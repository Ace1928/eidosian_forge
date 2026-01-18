from __future__ import annotations
from collections.abc import (
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
import pandas.core.common as com
from pandas.core.frame import _shared_docs
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import cartesian_product
from pandas.core.series import Series
def _compute_grand_margin(data: DataFrame, values, aggfunc, margins_name: Hashable='All'):
    if values:
        grand_margin = {}
        for k, v in data[values].items():
            try:
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)()
                elif isinstance(aggfunc, dict):
                    if isinstance(aggfunc[k], str):
                        grand_margin[k] = getattr(v, aggfunc[k])()
                    else:
                        grand_margin[k] = aggfunc[k](v)
                else:
                    grand_margin[k] = aggfunc(v)
            except TypeError:
                pass
        return grand_margin
    else:
        return {margins_name: aggfunc(data.index)}