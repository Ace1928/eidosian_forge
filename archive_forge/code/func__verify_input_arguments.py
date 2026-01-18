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
@staticmethod
def _verify_input_arguments(api_key: Optional[str], project: Optional[str], name: Optional[str], run: Optional[Union['Run', 'Handler']], neptune_run_kwargs: dict) -> None:
    if _NEPTUNE_AVAILABLE:
        from neptune import Run
        from neptune.handler import Handler
    else:
        from neptune.new import Run
        from neptune.new.handler import Handler
    if run is not None and (not isinstance(run, (Run, Handler))):
        raise ValueError('Run parameter expected to be of type `neptune.Run`, or `neptune.handler.Handler`.')
    any_neptune_init_arg_passed = any((arg is not None for arg in [api_key, project, name])) or neptune_run_kwargs
    if run is not None and any_neptune_init_arg_passed:
        raise ValueError("When an already initialized run object is provided, you can't provide other `neptune.init_run()` parameters.")