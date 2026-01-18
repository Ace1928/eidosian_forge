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
class TransformNode(DFAlgNode):
    """
    A node to represent a projection of a single frame.

    Provides expressions to compute each column of the projection.

    Parameters
    ----------
    base : HdkOnNativeDataframe
        A transformed frame.
    exprs : dict
        Expressions for frame's columns computation.
    fold : bool

    Attributes
    ----------
    input : list of HdkOnNativeDataframe
        Holds a single projected frame.
    exprs : dict
        Expressions used to compute frame's columns.
    """

    def __init__(self, base: 'HdkOnNativeDataframe', exprs: Dict[str, Union[InputRefExpr, LiteralExpr, OpExpr]], fold: bool=True):
        if fold and isinstance(base._op, TransformNode):
            self.input = [base._op.input[0]]
            self.exprs = exprs = translate_exprs_to_base(exprs, self.input[0])
            for col, expr in exprs.items():
                exprs[col] = expr.fold()
        else:
            self.input = [base]
            self.exprs = exprs

    @_inherit_docstrings(DFAlgNode.can_execute_hdk)
    def can_execute_hdk(self) -> bool:
        return self._check_exprs('can_execute_hdk')

    @_inherit_docstrings(DFAlgNode.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return self._check_exprs('can_execute_arrow')

    def execute_arrow(self, table: pa.Table) -> pa.Table:
        """
        Perform column selection on the frame using Arrow API.

        Parameters
        ----------
        table : pa.Table

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        cols = [expr.execute_arrow(table) for expr in self.exprs.values()]
        names = [ColNameCodec.encode(c) for c in self.exprs]
        return pa.table(cols, names)

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        TransformNode
        """
        return TransformNode(self.input[0], self.exprs)

    def is_simple_select(self):
        """
        Check if transform node is a simple selection.

        Simple selection can only use InputRefExpr expressions.

        Returns
        -------
        bool
            True for simple select and False otherwise.
        """
        return all((isinstance(expr, InputRefExpr) for expr in self.exprs.values()))

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
        res = f'{prefix}TransformNode:\n'
        for k, v in self.exprs.items():
            res += f'{prefix}  {k}: {v}\n'
        res += self._prints_input(prefix + '  ')
        return res

    def _check_exprs(self, attr) -> bool:
        """
        Check if the specified attribute is True for all expressions.

        Parameters
        ----------
        attr : str

        Returns
        -------
        bool
        """
        stack = list(self.exprs.values())
        while stack:
            expr = stack.pop()
            if not getattr(expr, attr)():
                return False
            if isinstance(expr, OpExpr):
                stack.extend(expr.operands)
        return True