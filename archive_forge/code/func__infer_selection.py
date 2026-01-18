from __future__ import annotations
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import (
@final
def _infer_selection(self, key, subset: Series | DataFrame):
    """
        Infer the `selection` to pass to our constructor in _gotitem.
        """
    selection = None
    if subset.ndim == 2 and (lib.is_scalar(key) and key in subset or lib.is_list_like(key)):
        selection = key
    elif subset.ndim == 1 and lib.is_scalar(key) and (key == subset.name):
        selection = key
    return selection