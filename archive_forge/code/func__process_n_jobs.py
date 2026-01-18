import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType,
from .callback import _EvalResultDict, record_evaluation
from .compat import (SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
from .engine import train
def _process_n_jobs(self, n_jobs: Optional[int]) -> int:
    """Convert special values of n_jobs to their actual values according to the formulas that apply.

        Parameters
        ----------
        n_jobs : int or None
            The original value of n_jobs, potentially having special values such as 'None' or
            negative integers.

        Returns
        -------
        n_jobs : int
            The value of n_jobs with special values converted to actual number of threads.
        """
    if n_jobs is None:
        n_jobs = _LGBMCpuCount(only_physical_cores=True)
    elif n_jobs < 0:
        n_jobs = max(_LGBMCpuCount(only_physical_cores=False) + 1 + n_jobs, 1)
    return n_jobs