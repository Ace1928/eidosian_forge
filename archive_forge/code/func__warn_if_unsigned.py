import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def _warn_if_unsigned(dtype):
    if np.issubdtype(dtype, np.unsignedinteger):
        ErrorMessage.single_warning('HDK does not support unsigned integer types, such types will be rounded up to the signed equivalent.')