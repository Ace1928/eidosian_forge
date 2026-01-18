from __future__ import annotations
import pickle as pkl
import re
import warnings
from typing import TYPE_CHECKING, Any, Hashable, Literal, Optional, Sequence, Union
import numpy as np
import pandas
import pandas.core.generic
import pandas.core.resample
import pandas.core.window.rolling
from pandas._libs import lib
from pandas._libs.tslibs import to_offset
from pandas._typing import (
from pandas.compat import numpy as numpy_compat
from pandas.core.common import count_not_none, pipe
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import ensure_index
from pandas.core.methods.describe import _refine_percentiles
from pandas.util._validators import (
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.accessor import CachedAccessor, ModinAPI
from modin.pandas.utils import is_scalar
from modin.utils import _inherit_docstrings, expanduser_path_arg, try_cast_to_pandas
from .utils import _doc_binary_op, is_full_grab_slice
def _copy_index_metadata(self, source, destination):
    """
        Copy Index metadata from `source` to `destination` inplace.
        """
    if hasattr(source, 'name') and hasattr(destination, 'name'):
        destination.name = source.name
    if hasattr(source, 'names') and hasattr(destination, 'names'):
        destination.names = source.names
    return destination