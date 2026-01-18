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
class LightningArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for pytorch-lightning."""

    def __init__(self, *args: Any, description: str='Lightning Trainer command line tool', env_prefix: str='PL', default_env: bool=False, **kwargs: Any) -> None:
        """Initialize argument parser that supports configuration file input.

        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.ArgumentParser.__init__>`_.

        Args:
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables. Set ``default_env=True`` to enable env parsing.
            default_env: Whether to parse environment variables.

        """
        if not _JSONARGPARSE_SIGNATURES_AVAILABLE:
            raise ModuleNotFoundError(f'{_JSONARGPARSE_SIGNATURES_AVAILABLE}')
        super().__init__(*args, description=description, env_prefix=env_prefix, default_env=default_env, **kwargs)
        self.callback_keys: List[str] = []
        self._optimizers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}
        self._lr_schedulers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}

    def add_lightning_class_args(self, lightning_class: Union[Callable[..., Union[Trainer, LightningModule, LightningDataModule, Callback]], Type[Trainer], Type[LightningModule], Type[LightningDataModule], Type[Callback]], nested_key: str, subclass_mode: bool=False, required: bool=True) -> List[str]:
        """Adds arguments from a lightning class to a nested key of the parser.

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.

        Returns:
            A list with the names of the class arguments added.

        """
        if callable(lightning_class) and (not isinstance(lightning_class, type)):
            lightning_class = class_from_function(lightning_class)
        if isinstance(lightning_class, type) and issubclass(lightning_class, (Trainer, LightningModule, LightningDataModule, Callback)):
            if issubclass(lightning_class, Callback):
                self.callback_keys.append(nested_key)
            if subclass_mode:
                return self.add_subclass_arguments(lightning_class, nested_key, fail_untyped=False, required=required)
            return self.add_class_arguments(lightning_class, nested_key, fail_untyped=False, instantiate=not issubclass(lightning_class, Trainer), sub_configs=True)
        raise MisconfigurationException(f'Cannot add arguments from: {lightning_class}. You should provide either a callable or a subclass of: Trainer, LightningModule, LightningDataModule, or Callback.')

    def add_optimizer_args(self, optimizer_class: Union[Type[Optimizer], Tuple[Type[Optimizer], ...]]=(Optimizer,), nested_key: str='optimizer', link_to: str='AUTOMATIC') -> None:
        """Adds arguments from an optimizer class to a nested key of the parser.

        Args:
            optimizer_class: Any subclass of :class:`torch.optim.Optimizer`. Use tuple to allow subclasses.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.

        """
        if isinstance(optimizer_class, tuple):
            assert all((issubclass(o, Optimizer) for o in optimizer_class))
        else:
            assert issubclass(optimizer_class, Optimizer)
        kwargs: Dict[str, Any] = {'instantiate': False, 'fail_untyped': False, 'skip': {'params'}}
        if isinstance(optimizer_class, tuple):
            self.add_subclass_arguments(optimizer_class, nested_key, **kwargs)
        else:
            self.add_class_arguments(optimizer_class, nested_key, sub_configs=True, **kwargs)
        self._optimizers[nested_key] = (optimizer_class, link_to)

    def add_lr_scheduler_args(self, lr_scheduler_class: Union[LRSchedulerType, Tuple[LRSchedulerType, ...]]=LRSchedulerTypeTuple, nested_key: str='lr_scheduler', link_to: str='AUTOMATIC') -> None:
        """Adds arguments from a learning rate scheduler class to a nested key of the parser.

        Args:
            lr_scheduler_class: Any subclass of ``torch.optim.lr_scheduler.{_LRScheduler, ReduceLROnPlateau}``. Use
                tuple to allow subclasses.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.

        """
        if isinstance(lr_scheduler_class, tuple):
            assert all((issubclass(o, LRSchedulerTypeTuple) for o in lr_scheduler_class))
        else:
            assert issubclass(lr_scheduler_class, LRSchedulerTypeTuple)
        kwargs: Dict[str, Any] = {'instantiate': False, 'fail_untyped': False, 'skip': {'optimizer'}}
        if isinstance(lr_scheduler_class, tuple):
            self.add_subclass_arguments(lr_scheduler_class, nested_key, **kwargs)
        else:
            self.add_class_arguments(lr_scheduler_class, nested_key, sub_configs=True, **kwargs)
        self._lr_schedulers[nested_key] = (lr_scheduler_class, link_to)