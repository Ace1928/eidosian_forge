from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
class StdAggregate(CompoundAggregate):
    """
        A sample standard deviation aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : list of BaseExpr
            An aggregated value.
        """

    def __init__(self, builder, arg):
        assert isinstance(arg[0], InputRefExpr)
        super().__init__(builder, arg[0])
        self._quad_name = self._arg.column + '__quad__'
        self._sum_name = self._arg.column + '__sum__'
        self._quad_sum_name = self._arg.column + '__quad_sum__'
        self._count_name = self._arg.column + '__count__'

    def gen_proj_exprs(self):
        """
            Generate values required for intermediate aggregates computation.

            Returns
            -------
            dict
                New column expressions mapped to their names.
            """
        expr = self._builder._translate(self._arg.mul(self._arg))
        return {self._quad_name: expr}

    def gen_agg_exprs(self):
        """
            Generate intermediate aggregates required for a compound aggregate computation.

            Returns
            -------
            dict
                New aggregate expressions mapped to their names.
            """
        count_expr = self._builder._translate(AggregateExpr('count', self._arg))
        sum_expr = self._builder._translate(AggregateExpr('sum', self._arg))
        self._sum_dtype = sum_expr._dtype
        qsum_expr = AggregateExpr('SUM', self._builder._ref_idx(self._arg.modin_frame, self._quad_name), dtype=sum_expr._dtype)
        return {self._sum_name: sum_expr, self._quad_sum_name: qsum_expr, self._count_name: count_expr}

    def gen_reduce_expr(self):
        """
            Generate an expression for a compound aggregate.

            Returns
            -------
            BaseExpr
                A final compound aggregate expression.
            """
        count_expr = self._builder._ref(self._arg.modin_frame, self._count_name)
        count_expr._dtype = _get_dtype(int)
        sum_expr = self._builder._ref(self._arg.modin_frame, self._sum_name)
        sum_expr._dtype = self._sum_dtype
        qsum_expr = self._builder._ref(self._arg.modin_frame, self._quad_sum_name)
        qsum_expr._dtype = self._sum_dtype
        null_expr = LiteralExpr(None)
        count_or_null = build_if_then_else(count_expr.eq(LiteralExpr(0)), null_expr, count_expr, count_expr._dtype)
        count_m_1_or_null = build_if_then_else(count_expr.eq(LiteralExpr(1)), null_expr, count_expr.sub(LiteralExpr(1)), count_expr._dtype)
        return qsum_expr.sub(sum_expr.mul(sum_expr).truediv(count_or_null)).truediv(count_m_1_or_null).pow(LiteralExpr(0.5))