import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union
from torch import Tensor
from typing_extensions import override
from lightning_fabric.loggers.logger import Logger, rank_zero_experiment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.logger import _add_prefix
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.types import _PATH
def _handle_value(value: Union[Tensor, Any]) -> Any:
    if isinstance(value, Tensor):
        return value.item()
    return value