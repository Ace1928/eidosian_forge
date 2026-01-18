from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def _std_diff_prop(p1, p2, ratio=1):
    return np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)