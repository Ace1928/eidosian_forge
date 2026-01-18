import numpy as np
import pandas as pd
import param
from packaging.version import Version
from ..core import Element, Operation
from ..core.data import PandasInterface
from ..core.util import _PANDAS_FUNC_LOOKUP, pandas_version
from ..element import Scatter
class RollingBase(param.Parameterized):
    """
    Parameters shared between `rolling` and `rolling_outlier_std`.
    """
    center = param.Boolean(default=True, doc='\n        Whether to set the x-coordinate at the center or right edge\n        of the window.')
    min_periods = param.Integer(default=None, allow_None=True, doc='\n       Minimum number of observations in window required to have a\n       value (otherwise result is NaN).')
    rolling_window = param.Integer(default=10, doc='\n        The window size over which to operate.')

    def _roll_kwargs(self):
        return {'window': self.p.rolling_window, 'center': self.p.center, 'min_periods': self.p.min_periods}