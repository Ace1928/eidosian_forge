from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _idx(self, frame, col):
    """
            Get a numeric input index for an input column.

            Parameters
            ----------
            frame : DFAlgNode
                An input frame.
            col : str
                An input column.

            Returns
            -------
            int
            """
    assert frame in self.input_offsets, f'unexpected reference to {frame.id_str()}'
    offs = self.input_offsets[frame]
    if frame in self.replacements:
        return self.replacements[frame].index(col) + offs
    if col == ColNameCodec.ROWID_COL_NAME:
        if not isinstance(self.frame_to_node[frame], CalciteScanNode):
            raise NotImplementedError('rowid can be accessed in materialized frames only')
        return len(frame._table_cols) + offs
    assert col in frame._table_cols, f"unexpected reference to '{col}' in {frame.id_str()}"
    return frame._table_cols.index(col) + offs