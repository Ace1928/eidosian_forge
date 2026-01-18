import re
from typing import Hashable, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow
from pandas._libs.lib import no_default
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import Index, MultiIndex, RangeIndex
from pyarrow.types import is_dictionary
from modin.core.dataframe.base.dataframe.utils import (
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.dataframe.pandas.metadata.dtypes import get_categories_dtype
from modin.core.dataframe.pandas.utils import concatenate
from modin.error_message import ErrorMessage
from modin.experimental.core.storage_formats.hdk.query_compiler import (
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from ..db_worker import DbTable
from ..df_algebra import (
from ..expr import (
from ..partitioning.partition_manager import HdkOnNativeDataframePartitionManager
from .utils import (
def _execute_arrow(self):
    """
        Compute the frame data using Arrow API.

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
    result = None
    stack = [self]
    while stack:
        frame = stack.pop()
        if callable(frame):
            if isinstance((result := frame(result)), DbTable):
                result = result.to_arrow()
        elif (input := getattr(frame._op, 'input', None)):
            if len(input) == 1:
                stack.append(frame._op.execute_arrow)
                stack.append(input[0])
            else:

                def to_arrow(result, op=frame._op, tables=[], frames=iter(input)):
                    """
                        Convert the input list to a list of arrow tables.

                        This function is created for each input list. When the function
                        is created, the frames iterator is saved in the `frames` argument.
                        Then, the function is added to the stack followed by the first
                        frame from the `frames` iterator. When the frame is processed, the
                        arrow table is added to the `tables` list. This procedure is
                        repeated until the iterator is not empty. When all the frames are
                        processed, the arrow tables are passed to `execute_arrow` and the
                        result is returned.
                        """
                    if (f := next(frames, None)) is None:
                        return op.execute_arrow(tables)
                    else:
                        stack.append(frame if callable(frame) else to_arrow)
                        stack.append(tables.append)
                        stack.append(f)
                        return result
                to_arrow(result)
        elif isinstance((result := frame._op.execute_arrow(result)), DbTable):
            result = result.to_arrow()
    return result