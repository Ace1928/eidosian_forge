import logging
import os
import re
import tempfile
from argparse import Namespace
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional, Union
import yaml
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.loggers.utilities import _scan_checkpoints
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
def _get_resolve_tags() -> Callable:
    from mlflow.tracking import context
    if hasattr(context, 'resolve_tags'):
        from mlflow.tracking.context import resolve_tags
    elif hasattr(context, 'registry'):
        from mlflow.tracking.context.registry import resolve_tags
    else:
        resolve_tags = lambda tags: tags
    return resolve_tags