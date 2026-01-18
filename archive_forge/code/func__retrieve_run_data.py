import contextlib
import logging
import os
from argparse import Namespace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Union
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.rank_zero import rank_zero_only
def _retrieve_run_data(self) -> None:
    if _NEPTUNE_AVAILABLE:
        from neptune.handler import Handler
    else:
        from neptune.new.handler import Handler
    assert self._run_instance is not None
    root_obj = self._run_instance
    if isinstance(root_obj, Handler):
        root_obj = root_obj.get_root_object()
    root_obj.wait()
    if root_obj.exists('sys/id'):
        self._run_short_id = root_obj['sys/id'].fetch()
        self._run_name = root_obj['sys/name'].fetch()
    else:
        self._run_short_id = 'OFFLINE'
        self._run_name = 'offline-name'