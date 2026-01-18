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
def _record_new_keys(self) -> Set[str]:
    """Records new keys that have not been logged before."""
    current_keys = set().union(*self.metrics)
    new_keys = current_keys - set(self.metrics_keys)
    self.metrics_keys.extend(new_keys)
    self.metrics_keys.sort()
    return new_keys