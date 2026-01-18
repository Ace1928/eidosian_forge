import operator
import socket
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse
import numpy as np
import scipy.sparse as ss
from .basic import LightGBMError, _choose_param_value, _ConfigAliases, _log_info, _log_warning
from .compat import (DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED, Client, Future, LGBMNotFittedError, concat,
from .sklearn import (LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor, _LGBM_ScikitCustomObjectiveFunction,
def _pad_eval_names(lgbm_model: LGBMModel, required_names: List[str]) -> LGBMModel:
    """Append missing (key, value) pairs to a LightGBM model's evals_result_ and best_score_ OrderedDict attrs based on a set of required eval_set names.

    Allows users to rely on expected eval_set names being present when fitting DaskLGBM estimators with ``eval_set``.
    """
    for eval_name in required_names:
        if eval_name not in lgbm_model.evals_result_:
            lgbm_model.evals_result_[eval_name] = {}
        if eval_name not in lgbm_model.best_score_:
            lgbm_model.best_score_[eval_name] = {}
    return lgbm_model