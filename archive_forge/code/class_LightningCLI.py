import os
import sys
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
class LightningCLI:
    """Implementation of a configurable command line tool for pytorch-lightning."""

    def __init__(self, model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]]=None, datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]]=None, save_config_callback: Optional[Type[SaveConfigCallback]]=SaveConfigCallback, save_config_kwargs: Optional[Dict[str, Any]]=None, trainer_class: Union[Type[Trainer], Callable[..., Trainer]]=Trainer, trainer_defaults: Optional[Dict[str, Any]]=None, seed_everything_default: Union[bool, int]=True, parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]=None, subclass_mode_model: bool=False, subclass_mode_data: bool=False, args: ArgsType=None, run: bool=True, auto_configure_optimizers: bool=True) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which are
        called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``parser_kwargs={"default_env":
        True}``. A full configuration yaml would be parsed from ``PL_CONFIG`` if set. Individual settings are so parsed
        from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <lightning-cli>`.

        Args:
            model_class: An optional :class:`~pytorch_lightning.core.LightningModule` class to train on or a
                callable which returns a :class:`~pytorch_lightning.core.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the config.
            save_config_kwargs: Parameters that will be used to instantiate the save_config_callback.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~pytorch_lightning.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <lightning-cli>`.
            seed_everything_default: Number for the :func:`~lightning_fabric.utilities.seed.seed_everything`
                seed value. Set to True to automatically choose a seed value.
                Setting it to False will avoid calling ``seed_everything``.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``. Command line style
                arguments can be given in a ``list``. Alternatively, structured config options can be given in a
                ``dict`` or ``jsonargparse.Namespace``.
            run: Whether subcommands should be added to run a :class:`~pytorch_lightning.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.

        """
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default
        self.parser_kwargs = parser_kwargs or {}
        self.auto_configure_optimizers = auto_configure_optimizers
        self.model_class = model_class
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = model_class is None or subclass_mode_model
        self.datamodule_class = datamodule_class
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = datamodule_class is None or subclass_mode_data
        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(self.parser_kwargs)
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser, args)
        self.subcommand = self.config['subcommand'] if run else None
        self._set_seed()
        self.before_instantiate_classes()
        self.instantiate_classes()
        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    def _setup_parser_kwargs(self, parser_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        subcommand_names = self.subcommands().keys()
        main_kwargs = {k: v for k, v in parser_kwargs.items() if k not in subcommand_names}
        subparser_kwargs = {k: v for k, v in parser_kwargs.items() if k in subcommand_names}
        return (main_kwargs, subparser_kwargs)

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault('dump_header', [f'pytorch_lightning=={pl.__version__}'])
        parser = LightningArgumentParser(**kwargs)
        parser.add_argument('-c', '--config', action=ActionConfigFile, help='Path to a configuration file in json or yaml format.')
        return parser

    def setup_parser(self, add_subcommands: bool, main_kwargs: Dict[str, Any], subparser_kwargs: Dict[str, Any]) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        if add_subcommands:
            self._subcommand_method_arguments: Dict[str, List[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument('--seed_everything', type=Union[bool, int], default=self.seed_everything_default, help='Set to an int to run seed_everything with this value before classes instantiation.Set to True to use a random seed.')

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        parser.add_lightning_class_args(self.trainer_class, 'trainer')
        trainer_defaults = {'trainer.' + k: v for k, v in self.trainer_defaults.items() if k != 'callbacks'}
        parser.set_defaults(trainer_defaults)
        parser.add_lightning_class_args(self._model_class, 'model', subclass_mode=self.subclass_mode_model)
        if self.datamodule_class is not None:
            parser.add_lightning_class_args(self._datamodule_class, 'data', subclass_mode=self.subclass_mode_data)
        else:
            parser.add_lightning_class_args(self._datamodule_class, 'data', subclass_mode=self.subclass_mode_data, required=False)

    def _add_arguments(self, parser: LightningArgumentParser) -> None:
        self.add_default_arguments_to_parser(parser)
        self.add_core_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)
        if self.auto_configure_optimizers:
            if not parser._optimizers:
                parser.add_optimizer_args((Optimizer,))
            if not parser._lr_schedulers:
                parser.add_lr_scheduler_args(LRSchedulerTypeTuple)
        self.link_optimizers_and_lr_schedulers(parser)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Implement to add extra arguments to the parser or link arguments.

        Args:
            parser: The parser object to which arguments can be added

        """

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {'fit': {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'}, 'validate': {'model', 'dataloaders', 'datamodule'}, 'test': {'model', 'dataloaders', 'datamodule'}, 'predict': {'model', 'dataloaders', 'datamodule'}}

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:
        """Adds subcommands to the input parser."""
        self._subcommand_parsers: Dict[str, LightningArgumentParser] = {}
        parser_subcommands = parser.add_subcommands()
        trainer_class = self.trainer_class if isinstance(self.trainer_class, type) else class_from_function(self.trainer_class)
        for subcommand in self.subcommands():
            fn = getattr(trainer_class, subcommand)
            description = _get_short_description(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault('description', description)
            subcommand_parser = self._prepare_subcommand_parser(trainer_class, subcommand, **subparser_kwargs)
            self._subcommand_parsers[subcommand] = subcommand_parser
            parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _prepare_subcommand_parser(self, klass: Type, subcommand: str, **kwargs: Any) -> LightningArgumentParser:
        parser = self.init_parser(**kwargs)
        self._add_arguments(parser)
        skip: Set[Union[str, int]] = set(self.subcommands()[subcommand])
        added = parser.add_method_arguments(klass, subcommand, skip=skip)
        self._subcommand_method_arguments[subcommand] = added
        return parser

    @staticmethod
    def link_optimizers_and_lr_schedulers(parser: LightningArgumentParser) -> None:
        """Creates argument links for optimizers and learning rate schedulers that specified a ``link_to``."""
        optimizers_and_lr_schedulers = {**parser._optimizers, **parser._lr_schedulers}
        for key, (class_type, link_to) in optimizers_and_lr_schedulers.items():
            if link_to == 'AUTOMATIC':
                continue
            if isinstance(class_type, tuple):
                parser.link_arguments(key, link_to)
            else:
                add_class_path = _add_class_path_generator(class_type)
                parser.link_arguments(key, link_to, compute_fn=add_class_path)

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        if args is not None and len(sys.argv) > 1:
            rank_zero_warn(f"LightningCLI's args parameter is intended to run from within Python like if it were from the command line. To prevent mistakes it is not recommended to provide both args and command line arguments, got: sys.argv[1:]={sys.argv[1:]}, args={args}.")
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, 'data')
        self.model = self._get(self.config_init, 'model')
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.

        """
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, 'trainer', default={}), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def _instantiate_trainer(self, config: Dict[str, Any], callbacks: List[Callback]) -> Trainer:
        key = 'callbacks'
        if key in config:
            if config[key] is None:
                config[key] = []
            elif not isinstance(config[key], list):
                config[key] = [config[key]]
            config[key].extend(callbacks)
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                config[key] += value if isinstance(value, list) else [value]
            if self.save_config_callback and (not config.get('fast_dev_run', False)):
                config_callback = self.save_config_callback(self._parser(self.subcommand), self.config.get(str(self.subcommand), self.config), **self.save_config_kwargs)
                config[key].append(config_callback)
        else:
            rank_zero_warn(f'The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will not be included.')
        return self.trainer_class(**config)

    def _parser(self, subcommand: Optional[str]) -> LightningArgumentParser:
        if subcommand is None:
            return self.parser
        return self._subcommand_parsers[subcommand]

    @staticmethod
    def configure_optimizers(lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion]=None) -> Any:
        """Override to customize the :meth:`~pytorch_lightning.core.LightningModule.configure_optimizers` method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).

        """
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': lr_scheduler.monitor}}
        return ([optimizer], [lr_scheduler])

    def _add_configure_optimizers_method_to_model(self, subcommand: Optional[str]) -> None:
        """Overrides the model's :meth:`~pytorch_lightning.core.LightningModule.configure_optimizers` method if a
        single optimizer and optionally a scheduler argument groups are added to the parser as 'AUTOMATIC'."""
        if not self.auto_configure_optimizers:
            return
        parser = self._parser(subcommand)

        def get_automatic(class_type: Union[Type, Tuple[Type, ...]], register: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]]) -> List[str]:
            automatic = []
            for key, (base_class, link_to) in register.items():
                if not isinstance(base_class, tuple):
                    base_class = (base_class,)
                if link_to == 'AUTOMATIC' and any((issubclass(c, class_type) for c in base_class)):
                    automatic.append(key)
            return automatic
        optimizers = get_automatic(Optimizer, parser._optimizers)
        lr_schedulers = get_automatic(LRSchedulerTypeTuple, parser._lr_schedulers)
        if len(optimizers) == 0:
            return
        if len(optimizers) > 1 or len(lr_schedulers) > 1:
            raise MisconfigurationException(f"`{self.__class__.__name__}.add_configure_optimizers_method_to_model` expects at most one optimizer and one lr_scheduler to be 'AUTOMATIC', but found {optimizers + lr_schedulers}. In this case the user is expected to link the argument groups and implement `configure_optimizers`, see https://lightning.ai/docs/pytorch/stable/common/lightning_cli.html#optimizers-and-learning-rate-schedulers")
        optimizer_class = parser._optimizers[optimizers[0]][0]
        optimizer_init = self._get(self.config_init, optimizers[0])
        if not isinstance(optimizer_class, tuple):
            optimizer_init = _global_add_class_path(optimizer_class, optimizer_init)
        if not optimizer_init:
            return
        lr_scheduler_init = None
        if lr_schedulers:
            lr_scheduler_class = parser._lr_schedulers[lr_schedulers[0]][0]
            lr_scheduler_init = self._get(self.config_init, lr_schedulers[0])
            if not isinstance(lr_scheduler_class, tuple):
                lr_scheduler_init = _global_add_class_path(lr_scheduler_class, lr_scheduler_init)
        if is_overridden('configure_optimizers', self.model):
            _warn(f'`{self.model.__class__.__name__}.configure_optimizers` will be overridden by `{self.__class__.__name__}.configure_optimizers`.')
        optimizer = instantiate_class(self.model.parameters(), optimizer_init)
        lr_scheduler = instantiate_class(optimizer, lr_scheduler_init) if lr_scheduler_init else None
        fn = partial(self.configure_optimizers, optimizer=optimizer, lr_scheduler=lr_scheduler)
        update_wrapper(fn, self.configure_optimizers)
        self.model.configure_optimizers = MethodType(fn, self.model)

    def _get(self, config: Namespace, key: str, default: Optional[Any]=None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _run_subcommand(self, subcommand: str) -> None:
        """Run the chosen subcommand."""
        before_fn = getattr(self, f'before_{subcommand}', None)
        if callable(before_fn):
            before_fn()
        default = getattr(self.trainer, subcommand)
        fn = getattr(self, subcommand, default)
        fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
        fn(**fn_kwargs)
        after_fn = getattr(self, f'after_{subcommand}', None)
        if callable(after_fn):
            after_fn()

    def _prepare_subcommand_kwargs(self, subcommand: str) -> Dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]}
        fn_kwargs['model'] = self.model
        if self.datamodule is not None:
            fn_kwargs['datamodule'] = self.datamodule
        return fn_kwargs

    def _set_seed(self) -> None:
        """Sets the seed."""
        config_seed = self._get(self.config, 'seed_everything')
        if config_seed is False:
            return
        if config_seed is True:
            config_seed = seed_everything(workers=True)
        else:
            config_seed = seed_everything(config_seed, workers=True)
        if self.subcommand:
            self.config[self.subcommand]['seed_everything'] = config_seed
        else:
            self.config['seed_everything'] = config_seed