import numpy as np
import pandas as pd
import param
from packaging.version import Version
from ..core import Element, Operation
from ..core.data import PandasInterface
from ..core.util import _PANDAS_FUNC_LOOKUP, pandas_version
from ..element import Scatter
def _roll_kwargs(self):
    return {'window': self.p.rolling_window, 'center': self.p.center, 'min_periods': self.p.min_periods}