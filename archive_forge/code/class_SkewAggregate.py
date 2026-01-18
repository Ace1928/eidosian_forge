from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
class SkewAggregate(CompoundAggregate):
    """
        An unbiased skew aggregate generator.

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
        self._cube_name = self._arg.column + '__cube__'
        self._sum_name = self._arg.column + '__sum__'
        self._quad_sum_name = self._arg.column + '__quad_sum__'
        self._cube_sum_name = self._arg.column + '__cube_sum__'
        self._count_name = self._arg.column + '__count__'

    def gen_proj_exprs(self):
        """
            Generate values required for intermediate aggregates computation.

            Returns
            -------
            dict
                New column expressions mapped to their names.
            """
        quad_expr = self._builder._translate(self._arg.mul(self._arg))
        cube_expr = self._builder._translate(self._arg.mul(self._arg).mul(self._arg))
        return {self._quad_name: quad_expr, self._cube_name: cube_expr}

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
        csum_expr = AggregateExpr('SUM', self._builder._ref_idx(self._arg.modin_frame, self._cube_name), dtype=sum_expr._dtype)
        return {self._sum_name: sum_expr, self._quad_sum_name: qsum_expr, self._cube_sum_name: csum_expr, self._count_name: count_expr}

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
        csum_expr = self._builder._ref(self._arg.modin_frame, self._cube_sum_name)
        csum_expr._dtype = self._sum_dtype
        mean_expr = sum_expr.truediv(count_expr)
        part1 = count_expr.mul(count_expr.sub(LiteralExpr(1)).pow(LiteralExpr(0.5))).truediv(count_expr.sub(LiteralExpr(2)))
        part2 = csum_expr.sub(mean_expr.mul(qsum_expr).mul(LiteralExpr(3.0))).add(mean_expr.mul(mean_expr).mul(sum_expr).mul(LiteralExpr(2.0)))
        part3 = qsum_expr.sub(mean_expr.mul(sum_expr)).pow(LiteralExpr(1.5))
        skew_expr = part1.mul(part2).truediv(part3)
        return build_if_then_else(count_expr.le(LiteralExpr(2)), LiteralExpr(None), skew_expr, skew_expr._dtype)