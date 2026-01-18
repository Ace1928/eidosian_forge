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
class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.
        save_to_log_dir: Whether to save the config to the log_dir.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run

    """

    def __init__(self, parser: LightningArgumentParser, config: Namespace, config_filename: str='config.yaml', overwrite: bool=False, multifile: bool=False, save_to_log_dir: bool=True) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.save_to_log_dir = save_to_log_dir
        self.already_saved = False
        if not save_to_log_dir and (not is_overridden('save_config', self, SaveConfigCallback)):
            raise ValueError('`save_to_log_dir=False` only makes sense when subclassing SaveConfigCallback to implement `save_config` and it is desired to disable the standard behavior of saving to log_dir.')

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return
        if self.save_to_log_dir:
            log_dir = trainer.log_dir
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)
            if not self.overwrite:
                file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(f'{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting results of a previous run. You can delete the previous config file, set `LightningCLI(save_config_callback=None)` to disable config saving, or set `LightningCLI(save_config_kwargs={{"overwrite": True}})` to overwrite the config file.')
            if trainer.is_global_zero:
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile)
        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Implement to save the config in some other place additional to the standard log_dir.

        Example:
            def save_config(self, trainer, pl_module, stage):
                if isinstance(trainer.logger, Logger):
                    config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                    trainer.logger.log_hyperparams({"config": config})

        Note:
            This method is only called on rank zero. This allows to implement a custom save config without having to
            worry about ranks or race conditions. Since it only runs on rank zero, any collective call will make the
            process hang waiting for a broadcast. If you need to make collective calls, implement the setup method
            instead.

        """