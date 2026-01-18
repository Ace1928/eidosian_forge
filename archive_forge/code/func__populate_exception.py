import os
import pandas as pd
import pyarrow
from typing import Optional, Union
from ray.air.result import Result
from ray.cloudpickle import cloudpickle
from ray.exceptions import RayTaskError
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.error import TuneError
from ray.tune.experiment import Trial
from ray.util import PublicAPI
@staticmethod
def _populate_exception(trial: Trial) -> Optional[Union[TuneError, RayTaskError]]:
    if trial.status == Trial.TERMINATED:
        return None
    if trial.pickled_error_file and os.path.exists(trial.pickled_error_file):
        with open(trial.pickled_error_file, 'rb') as f:
            e = cloudpickle.load(f)
            return e
    elif trial.error_file and os.path.exists(trial.error_file):
        with open(trial.error_file, 'r') as f:
            return TuneError(f.read())
    return None