from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
class CompoundAggregate:
    """
        A base class for a compound aggregate translation.

        Translation is done in three steps. Step 1 is an additional
        values generation using a projection. Step 2 is a generation
        of aggregates that will be later used for a compound aggregate
        value computation. Step 3 is a final aggregate value generation
        using another projection.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : BaseExpr or List of BaseExpr
            An aggregated values.
        """

    def __init__(self, builder, arg):
        self._builder = builder
        self._arg = arg

    def gen_proj_exprs(self):
        """
            Generate values required for intermediate aggregates computation.

            Returns
            -------
            dict
                New column expressions mapped to their names.
            """
        return []

    def gen_agg_exprs(self):
        """
            Generate intermediate aggregates required for a compound aggregate computation.

            Returns
            -------
            dict
                New aggregate expressions mapped to their names.
            """
        pass

    def gen_reduce_expr(self):
        """
            Generate an expression for a compound aggregate.

            Returns
            -------
            BaseExpr
                A final compound aggregate expression.
            """
        pass