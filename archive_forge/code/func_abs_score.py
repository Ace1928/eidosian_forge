import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
@staticmethod
def abs_score(level=0.01):
    return ScoreConfig(score_type=ScoreType.Abs, multiplier=1, score_level=level)