from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal, TypedDict
from xarray.core.utils import FrozenDict
def _warn_on_setting_enable_cftimeindex(enable_cftimeindex):
    warnings.warn('The enable_cftimeindex option is now a no-op and will be removed in a future version of xarray.', FutureWarning)