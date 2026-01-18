import math
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .basic import Booster, _data_from_pandas, _is_zero, _log_warning, _MissingType
from .compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED, pd_DataFrame
from .sklearn import LGBMModel
def _determine_direction_for_categorical_split(fval: float, thresholds: str) -> str:
    if math.isnan(fval) or int(fval) < 0:
        return 'right'
    int_thresholds = {int(t) for t in thresholds.split('||')}
    return 'left' if int(fval) in int_thresholds else 'right'