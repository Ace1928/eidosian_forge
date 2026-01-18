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
def _split_to_parts(data: _DaskCollection, is_matrix: bool) -> List[_DaskPart]:
    parts = data.to_delayed()
    if isinstance(parts, np.ndarray):
        if is_matrix:
            assert parts.shape[1] == 1
        else:
            assert parts.ndim == 1 or parts.shape[1] == 1
        parts = parts.flatten().tolist()
    return parts