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
def _predict_part(part: _DaskPart, model: LGBMModel, raw_score: bool, pred_proba: bool, pred_leaf: bool, pred_contrib: bool, **kwargs: Any) -> _DaskPart:
    result: _DaskPart
    if part.shape[0] == 0:
        result = np.array([])
    elif pred_proba:
        result = model.predict_proba(part, raw_score=raw_score, pred_leaf=pred_leaf, pred_contrib=pred_contrib, **kwargs)
    else:
        result = model.predict(part, raw_score=raw_score, pred_leaf=pred_leaf, pred_contrib=pred_contrib, **kwargs)
    if isinstance(part, pd_DataFrame):
        if len(result.shape) == 2:
            result = pd_DataFrame(result, index=part.index)
        else:
            result = pd_Series(result, index=part.index, name='predictions')
    return result