import copy
import fnmatch
import io
import json
import logging
from numbers import Number
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow.fs
from ray.util.annotations import PublicAPI
from ray.air.constants import (
from ray.train import Checkpoint
from ray.train._internal.storage import (
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.tune.utils.serialization import TuneFunctionDecoder
from ray.tune.utils.util import is_nan_or_inf, is_nan, unflattened_lookup
@classmethod
def _find_newest_experiment_checkpoint(cls, fs: pyarrow.fs.FileSystem, experiment_fs_path: Union[str, os.PathLike]) -> Optional[str]:
    """Return the most recent experiment checkpoint path."""
    filenames = _list_at_fs_path(fs=fs, fs_path=experiment_fs_path)
    pattern = TuneController.CKPT_FILE_TMPL.format('*')
    matching = fnmatch.filter(filenames, pattern)
    if not matching:
        return None
    filename = max(matching)
    return os.path.join(experiment_fs_path, filename)