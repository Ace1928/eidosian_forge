import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
class BestRun(NamedTuple):
    """
    The best run found by a hyperparameter search (see [`~Trainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`Dict[str, Any]`):
            The hyperparameters picked to get this run.
        run_summary (`Optional[Any]`):
            A summary of tuning experiments. `ray.tune.ExperimentAnalysis` object for Ray backend.
    """
    run_id: str
    objective: Union[float, List[float]]
    hyperparameters: Dict[str, Any]
    run_summary: Optional[Any] = None