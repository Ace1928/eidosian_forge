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
@classmethod
def _is_zero_copy_arrow_op(cls, op) -> bool:
    """
        Check whether the passed node of the delayed computation tree could be executed zero-copy via PyArrow execution.

        Parameters
        ----------
        op : DFAlgNode

        Returns
        -------
        bool
        """
    is_zero_copy_op = False
    if isinstance(op, (FrameNode, TransformNode, UnionNode)):
        is_zero_copy_op = True
    elif isinstance(op, MaskNode) and (isinstance(op.row_positions, slice) or is_range_like(op.row_positions)):
        is_zero_copy_op = True
    return is_zero_copy_op and all((cls._is_zero_copy_arrow_op(_op) for _op in getattr(op, 'inputs', [])))