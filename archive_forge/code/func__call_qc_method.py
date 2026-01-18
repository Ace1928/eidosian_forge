from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import pandas.core.window.rolling
from pandas.core.dtypes.common import is_list_like
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import _inherit_docstrings
def _call_qc_method(self, method_name, *args, **kwargs):
    return self._aggregate(method_name, *args, **kwargs)._query_compiler