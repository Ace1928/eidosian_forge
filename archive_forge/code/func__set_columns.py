from functools import wraps
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import is_bool_dtype, is_integer_dtype
from modin.core.storage_formats import BaseQueryCompiler
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
def _set_columns(self, columns):
    """
        Set new columns.

        Parameters
        ----------
        columns : list-like
            New columns.
        """
    if self._modin_frame._has_unsupported_data:
        default_axis_setter(1)(self, columns)
    else:
        try:
            self._modin_frame = self._modin_frame._set_columns(columns)
        except NotImplementedError:
            default_axis_setter(1)(self, columns)
            self._modin_frame._has_unsupported_data = True