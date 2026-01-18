from __future__ import annotations
import warnings
from typing import Hashable, Iterable, Mapping, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import ArrayLike, DtypeBackend, Scalar, npt
from pandas.core.dtypes.common import is_list_like
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import enable_logging
from modin.pandas.io import to_pandas
from modin.utils import _inherit_docstrings
from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series
def _wrap_in_series_object(qc_result):
    if isinstance(qc_result, type(x._query_compiler)):
        return Series(query_compiler=qc_result)
    if isinstance(qc_result, (tuple, list)):
        return tuple([_wrap_in_series_object(result) for result in qc_result])
    return qc_result