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
def _retrieve_rows(self, metric: Optional[str]=None, mode: Optional[str]=None) -> Dict[str, Any]:
    assert mode is None or mode in ['max', 'min']
    assert not mode or metric
    rows = {}
    for path, df in self.trial_dataframes.items():
        if df.empty:
            continue
        if metric not in df:
            idx = -1
        elif mode == 'max':
            idx = df[metric].idxmax()
        elif mode == 'min':
            idx = df[metric].idxmin()
        else:
            idx = -1
        try:
            rows[path] = df.iloc[idx].to_dict()
        except TypeError:
            logger.warning('Warning: Non-numerical value(s) encountered for {}'.format(path))
    return rows