import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from lightning_fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
from lightning_fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities.logger import _convert_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
class ExperimentWriter(_FabricExperimentWriter):
    """Experiment writer for CSVLogger.

    Currently, supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    This logger supports logging to remote filesystems via ``fsspec``. Make sure you have it installed.

    Args:
        log_dir: Directory for the experiment logs

    """
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir=log_dir)
        self.hparams: Dict[str, Any] = {}

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    @override
    def save(self) -> None:
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)
        return super().save()