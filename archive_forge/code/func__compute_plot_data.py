from __future__ import annotations
from abc import (
from collections.abc import (
from typing import (
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
from pandas.plotting._matplotlib.tools import (
@final
def _compute_plot_data(self) -> None:
    data = self.data
    if self.by is not None:
        self.subplots = True
        data = reconstruct_data_with_by(self.data, by=self.by, cols=self.columns)
    data = data.infer_objects(copy=False)
    include_type = [np.number, 'datetime', 'datetimetz', 'timedelta']
    if self.include_bool is True:
        include_type.append(np.bool_)
    exclude_type = None
    if self._kind == 'box':
        include_type = [np.number]
        exclude_type = ['timedelta']
    if self._kind == 'scatter':
        include_type.extend(['object', 'category', 'string'])
    numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)
    is_empty = numeric_data.shape[-1] == 0
    if is_empty:
        raise TypeError('no numeric data to plot')
    self.data = numeric_data.apply(type(self)._convert_to_ndarray)