import logging
import os
import pickle
import sys
import threading
import warnings
from types import ModuleType, TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type
from packaging.version import Version
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.utilities.migration.migration import _migration_index
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _pl_migrate_checkpoint(checkpoint: _CHECKPOINT, checkpoint_path: Optional[_PATH]=None) -> _CHECKPOINT:
    """Applies Lightning version migrations to a checkpoint dictionary and prints infos for the user.

    This function is used by the Lightning Trainer when resuming from a checkpoint.

    """
    old_version = _get_version(checkpoint)
    checkpoint, migrations = migrate_checkpoint(checkpoint)
    new_version = _get_version(checkpoint)
    if not migrations or checkpoint_path is None:
        return checkpoint
    path_hint = os.path.relpath(checkpoint_path, os.getcwd()) if not _IS_WINDOWS else os.path.abspath(checkpoint_path)
    _log.info(f'Lightning automatically upgraded your loaded checkpoint from v{old_version} to v{new_version}. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint {str(path_hint)}`')
    return checkpoint