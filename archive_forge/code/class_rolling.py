import numpy as np
import pandas as pd
import param
from packaging.version import Version
from ..core import Element, Operation
from ..core.data import PandasInterface
from ..core.util import _PANDAS_FUNC_LOOKUP, pandas_version
from ..element import Scatter
class rolling(Operation, RollingBase):
    """
    Applies a function over a rolling window.
    """
    window_type = param.ObjectSelector(default=None, allow_None=True, objects=['boxcar', 'triang', 'blackman', 'hamming', 'bartlett', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann', 'kaiser', 'gaussian', 'general_gaussian', 'slepian'], doc='The shape of the window to apply')
    function = param.Callable(default=np.mean, doc='\n        The function to apply over the rolling window.')

    def _process_layer(self, element, key=None):
        xdim = element.kdims[0].name
        df = PandasInterface.as_dframe(element)
        df = df.set_index(xdim).rolling(win_type=self.p.window_type, **self._roll_kwargs())
        if self.p.window_type is None:
            kwargs = {'raw': True} if pandas_version >= Version('0.23.0') else {}
            rolled = df.apply(self.p.function, **kwargs)
        elif self.p.function is np.mean:
            rolled = df.mean()
        elif self.p.function is np.sum:
            rolled = df.sum()
        else:
            raise ValueError('Rolling window function only supports mean and sum when custom window_type is supplied')
        return element.clone(rolled.reset_index())

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)