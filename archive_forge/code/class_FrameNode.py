import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
class FrameNode(DFAlgNode):
    """
    A node to reference a materialized frame.

    Parameters
    ----------
    modin_frame : HdkOnNativeDataframe
        Referenced frame.

    Attributes
    ----------
    modin_frame : HdkOnNativeDataframe
        Referenced frame.
    """

    def __init__(self, modin_frame: 'HdkOnNativeDataframe'):
        self.modin_frame = modin_frame

    @_inherit_docstrings(DFAlgNode.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return self.modin_frame._has_arrow_table()

    def execute_arrow(self, ignore=None) -> Union[DbTable, pa.Table, pandas.DataFrame]:
        """
        Materialized frame.

        If `can_execute_arrow` returns True, this method returns an arrow table,
        otherwise - a pandas Dataframe or DbTable.

        Parameters
        ----------
        ignore : None, pa.Table or list of pa.Table, default: None

        Returns
        -------
        DbTable or pa.Table or pandas.Dataframe
        """
        frame = self.modin_frame
        if frame._partitions is not None:
            part = frame._partitions[0][0]
            to_arrow = part.raw and (not frame._has_unsupported_data)
            return part.get(to_arrow)
        if frame._has_unsupported_data:
            return pandas.DataFrame(index=frame._index_cache, columns=frame._columns_cache)
        if frame._index_cache or frame._columns_cache:
            return pa.Table.from_pandas(pandas.DataFrame(index=frame._index_cache, columns=frame._columns_cache))
        return EMPTY_ARROW_TABLE

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        FrameNode
        """
        return FrameNode(self.modin_frame)

    def _append_partitions(self, partitions):
        """
        Append all partitions of the referenced frame to `partitions` list.

        Parameters
        ----------
        partitions : list
            Output list of partitions.
        """
        partitions += self.modin_frame._partitions.flatten()

    def _append_frames(self, frames):
        """
        Append the referenced frame to `frames` list.

        Parameters
        ----------
        frames : list
            Output list of frames.
        """
        frames.append(self.modin_frame)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return f'{prefix}{self.modin_frame.id_str()}'