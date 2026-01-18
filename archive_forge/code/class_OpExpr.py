import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
class OpExpr(BaseExpr):
    """
    A generic operation expression.

    Used for arithmetic, comparisons, conditional operations, etc.

    Parameters
    ----------
    op : str
        Operation name.
    operands : list of BaseExpr
        Operation operands.
    dtype : dtype
        Result data type.

    Attributes
    ----------
    op : str
        Operation name.
    operands : list of BaseExpr
        Operation operands.
    _dtype : dtype
        Result data type.
    partition_keys : list of BaseExpr, optional
        This attribute is used with window functions only and contains
        a list of column expressions to partition the result set.
    order_keys : list of dict, optional
        This attribute is used with window functions only and contains
        order clauses.
    lower_bound : dict, optional
        Lover bound for windowed aggregates.
    upper_bound : dict, optional
        Upper bound for windowed aggregates.
    """
    _FOLD_OPS = {'+': lambda self: self._fold_arithm('__add__'), '-': lambda self: self._fold_arithm('__sub__'), '*': lambda self: self._fold_arithm('__mul__'), 'POWER': lambda self: self._fold_arithm('__pow__'), '/': lambda self: self._fold_arithm('__truediv__'), '//': lambda self: self._fold_arithm('__floordiv__'), 'BIT_NOT': lambda self: self._fold_invert(), 'CAST': lambda self: self._fold_literal('cast', self._dtype), 'IS NULL': lambda self: self._fold_literal('is_null'), 'IS NOT NULL': lambda self: self._fold_literal('is_not_null')}
    _ARROW_EXEC = {'+': lambda self, table: self._pc('add', table), '-': lambda self, table: self._pc('subtract', table), '*': lambda self, table: self._pc('multiply', table), 'POWER': lambda self, table: self._pc('power', table), '/': lambda self, table: self._pc('divide', table), '//': lambda self, table: self._pc('divide', table), 'BIT_NOT': lambda self, table: self._invert(table), 'CAST': lambda self, table: self._col(table).cast(to_arrow_type(self._dtype)), 'IS NULL': lambda self, table: self._col(table).is_null(nan_is_null=True), 'IS NOT NULL': lambda self, table: pc.invert(self._col(table).is_null(nan_is_null=True))}
    _UNSUPPORTED_HDK_OPS = {}

    def __init__(self, op, operands, dtype):
        self.op = op
        self.operands = operands
        self._dtype = dtype

    def set_window_opts(self, partition_keys, order_keys, order_ascending, na_pos):
        """
        Set the window function options.

        Parameters
        ----------
        partition_keys : list of BaseExpr
        order_keys : list of BaseExpr
        order_ascending : list of bool
        na_pos : {"FIRST", "LAST"}
        """
        self.is_rows = True
        self.partition_keys = partition_keys
        self.order_keys = []
        for key, asc in zip(order_keys, order_ascending):
            key = {'field': key, 'direction': 'ASCENDING' if asc else 'DESCENDING', 'nulls': na_pos}
            self.order_keys.append(key)
        self.lower_bound = {'unbounded': True, 'preceding': True, 'following': False, 'is_current_row': False, 'offset': None, 'order_key': 0}
        self.upper_bound = {'unbounded': False, 'preceding': False, 'following': False, 'is_current_row': True, 'offset': None, 'order_key': 1}

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        OpExpr
        """
        op = OpExpr(self.op, self.operands.copy(), self._dtype)
        if (pk := getattr(self, 'partition_keys', None)):
            op.partition_keys = pk
            op.is_rows = self.is_rows
            op.order_keys = self.order_keys
            op.lower_bound = self.lower_bound
            op.upper_bound = self.upper_bound
        return op

    @_inherit_docstrings(BaseExpr.nested_expressions)
    def nested_expressions(self) -> Generator[Type['BaseExpr'], Type['BaseExpr'], Type['BaseExpr']]:
        expr = (yield from super().nested_expressions())
        if (partition_keys := getattr(self, 'partition_keys', None)):
            for i, key in enumerate(partition_keys):
                new_key = (yield key)
                if new_key is not None:
                    if new_key is not key:
                        if expr is self:
                            expr = self.copy()
                        expr.partition_keys[i] = new_key
                    yield expr
            for i, key in enumerate(self.order_keys):
                field = key['field']
                new_field = (yield field)
                if new_field is not None:
                    if new_field is not field:
                        if expr is self:
                            expr = self.copy()
                        expr.order_keys[i]['field'] = new_field
                    yield expr
        return expr

    @_inherit_docstrings(BaseExpr.fold)
    def fold(self):
        super().fold()
        return self if (op := self._FOLD_OPS.get(self.op, None)) is None else op(self)

    def _fold_arithm(self, op) -> Union['OpExpr', LiteralExpr]:
        """
        Fold arithmetic expressions.

        Parameters
        ----------
        op : str

        Returns
        -------
        OpExpr or LiteralExpr
        """
        operands = self.operands
        i = 0
        while i < len(operands):
            if isinstance((o := operands[i]), OpExpr):
                if self.op == o.op:
                    operands[i:i + 1] = o.operands
                else:
                    i += 1
                    continue
            if i == 0:
                i += 1
                continue
            if isinstance(o, LiteralExpr) and isinstance(operands[i - 1], LiteralExpr):
                val = getattr(operands[i - 1].val, op)(o.val)
                operands[i - 1] = LiteralExpr(val).cast(o._dtype)
                del operands[i]
            else:
                i += 1
        return operands[0] if len(operands) == 1 else self

    def _fold_invert(self) -> Union['OpExpr', LiteralExpr]:
        """
        Fold invert expression.

        Returns
        -------
        OpExpr or LiteralExpr
        """
        assert len(self.operands) == 1
        op = self.operands[0]
        if isinstance(op, LiteralExpr):
            return LiteralExpr(~op.val, op._dtype)
        if isinstance(op, OpExpr):
            if op.op == 'IS NULL':
                return OpExpr('IS NOT NULL', op.operands, op._dtype)
            if op.op == 'IS NOT NULL':
                return OpExpr('IS NULL', op.operands, op._dtype)
        return self

    def _fold_literal(self, op, *args):
        """
        Fold literal expressions.

        Parameters
        ----------
        op : str

        *args : list

        Returns
        -------
        OpExpr or LiteralExpr
        """
        assert len(self.operands) == 1
        expr = self.operands[0]
        return getattr(expr, op)(*args) if isinstance(expr, LiteralExpr) else self

    @_inherit_docstrings(BaseExpr.can_execute_hdk)
    def can_execute_hdk(self) -> bool:
        return self.op not in self._UNSUPPORTED_HDK_OPS

    @_inherit_docstrings(BaseExpr.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return self.op in self._ARROW_EXEC

    @_inherit_docstrings(BaseExpr.execute_arrow)
    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        return self._ARROW_EXEC[self.op](self, table)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        if (pk := getattr(self, 'partition_keys', None)):
            return f'({self.op} {self.operands} {pk} {self.order_keys} [{self._dtype}])'
        return f'({self.op} {self.operands} [{self._dtype}])'

    def _col(self, table: pa.Table) -> pa.ChunkedArray:
        """
        Return the column referenced by the `InputRefExpr` operand.

        Parameters
        ----------
        table : pa.Table

        Returns
        -------
        pa.ChunkedArray
        """
        assert isinstance(self.operands[0], InputRefExpr)
        return self.operands[0].execute_arrow(table)

    def _pc(self, op: str, table: pa.Table) -> pa.ChunkedArray:
        """
        Perform the specified pyarrow.compute operation on the operands.

        Parameters
        ----------
        op : str
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray
        """
        op = getattr(pc, op)
        val = self._op_value(0, table)
        for i in range(1, len(self.operands)):
            val = op(val, self._op_value(i, table))
        if not isinstance(val, pa.ChunkedArray):
            val = LiteralExpr(val).execute_arrow(table)
        if val.type != (at := to_arrow_type(self._dtype)):
            val = val.cast(at)
        return val

    def _op_value(self, op_idx: int, table: pa.Table):
        """
        Get the specified operand value.

        Parameters
        ----------
        op_idx : int
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray or expr.val
        """
        expr = self.operands[op_idx]
        return expr.val if isinstance(expr, LiteralExpr) else expr.execute_arrow(table)

    def _invert(self, table: pa.Table) -> pa.ChunkedArray:
        """
        Bitwise inverse the column values.

        Parameters
        ----------
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray
        """
        if is_bool_dtype(self._dtype):
            return pc.invert(self._col(table))
        try:
            return pc.bit_wise_not(self._col(table))
        except pa.ArrowNotImplementedError as err:
            raise TypeError(str(err))