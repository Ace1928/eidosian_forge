from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Optional
import os
import tempfile
import warnings
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.utils import flatten_dict
from ray.util import log_once
from lightgbm.callback import CallbackEnv
from lightgbm.basic import Booster
from ray.util.annotations import Deprecated
@Deprecated
class TuneReportCallback(TuneReportCheckpointCallback):

    def __init__(self, metrics: Optional[Union[str, List[str], Dict[str, str]]]=None, results_postprocessing_fn: Optional[Callable[[Dict[str, Union[float, List[float]]]], Dict[str, float]]]=None):
        if log_once('tune_lightgbm_report_deprecated'):
            warnings.warn('`ray.tune.integration.lightgbm.TuneReportCallback` is deprecated. Use `ray.tune.integration.lightgbm.TuneCheckpointReportCallback` instead.')
        super().__init__(metrics=metrics, results_postprocessing_fn=results_postprocessing_fn, frequency=0)