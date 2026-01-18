import collections
from typing import Any, Dict, Iterable, Optional, Sequence
import numpy as np
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.error_message import ErrorMessage
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .column import HdkProtocolColumn
from .utils import raise_copy_alert_if_materialize
@property
def _chunk_slices(self) -> np.ndarray:
    """
        Compute chunks start-stop indices in the underlying PyArrow table.

        Returns
        -------
        np.ndarray
            An array holding start-stop indices of the chunks, for ex. ``[0, 5, 10, 20]``
            describes 3 chunks bound by the following indices:
                chunk1: [0, 5),
                chunk2: [5, 10),
                chunk3: [10, 20).

        Notes
        -----
        Arrow table allows for the columns to be chunked independently, so in order to satisfy
        the protocol's requirement of equally chunked columns, we have to align column chunks
        with the minimal one. For example:
            Originally chunked table:        Aligned table:
            |col0|col1|                       |col0|col1|
            |    |    |                       |    |    |
            |0   |a   |                       |0   |a   |
            |----|b   |                       |----|----|
            |1   |----|                       |1   |b   |
            |2   |c   |                       |----|----|
            |3   |d   |                       |2   |c   |
            |----|----|                       |3   |d   |
            |4   |e   |                       |----|----|
                                              |4   |e   |
        """
    if self.__chunk_slices is None:
        at = self._pyarrow_table
        col_slices = set({0})
        for col in at.columns:
            col_slices = col_slices.union(np.cumsum([len(chunk) for chunk in col.chunks]))
        self.__chunk_slices = np.sort(np.fromiter(col_slices, dtype=int, count=len(col_slices)))
    return self.__chunk_slices