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
def _set_legacy_version(checkpoint: _CHECKPOINT, version: str) -> None:
    """Set the legacy version of a Lightning checkpoint if a legacy version is not already set."""
    checkpoint.setdefault('legacy_pytorch-lightning_version', version)