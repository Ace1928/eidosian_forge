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
class pl_legacy_patch:
    """Registers legacy artifacts (classes, methods, etc.) that were removed but still need to be included for
    unpickling old checkpoints. The following patches apply.

        1. ``pytorch_lightning.utilities.argparse._gpus_arg_default``: Applies to all checkpoints saved prior to
           version 1.2.8. See: https://github.com/Lightning-AI/lightning/pull/6898
        2. ``pytorch_lightning.utilities.argparse_utils``: A module that was deprecated in 1.2 and removed in 1.4,
           but still needs to be available for import for legacy checkpoints.
        3. ``pytorch_lightning.utilities.enums._FaultTolerantMode``: This enum was removed in 2.0 but was pickled
           into older checkpoints.
        4. In legacy versions of Lightning, callback classes got pickled into the checkpoint. These classes have a
           module import path under ``pytorch_lightning`` and must be redirected to the ``pytorch_lightning``.

    Example:

        with pl_legacy_patch():
            torch.load("path/to/legacy/checkpoint.ckpt")

    """

    def __enter__(self) -> 'pl_legacy_patch':
        _lock.acquire()
        legacy_argparse_module = ModuleType('pytorch_lightning.utilities.argparse_utils')
        sys.modules['pytorch_lightning.utilities.argparse_utils'] = legacy_argparse_module
        legacy_argparse_module._gpus_arg_default = lambda x: x
        pl.utilities.argparse._gpus_arg_default = lambda x: x

        class _FaultTolerantMode(LightningEnum):
            DISABLED = 'disabled'
            AUTOMATIC = 'automatic'
            MANUAL = 'manual'
        pl.utilities.enums._FaultTolerantMode = _FaultTolerantMode
        self._old_unpickler = pickle.Unpickler
        pickle.Unpickler = _RedirectingUnpickler
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], exc_traceback: Optional[TracebackType]) -> None:
        if hasattr(pl.utilities.argparse, '_gpus_arg_default'):
            delattr(pl.utilities.argparse, '_gpus_arg_default')
        del sys.modules['pytorch_lightning.utilities.argparse_utils']
        if hasattr(pl.utilities.enums, '_FaultTolerantMode'):
            delattr(pl.utilities.enums, '_FaultTolerantMode')
        pickle.Unpickler = self._old_unpickler
        _lock.release()