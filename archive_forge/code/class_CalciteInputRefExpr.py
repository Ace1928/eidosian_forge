import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteInputRefExpr(BaseExpr):
    """
    Calcite version of input column reference.

    Calcite translation should replace all ``InputRefExpr``.

    Calcite references columns by their indexes (positions in input table).
    If there are multiple input tables for Calcite node, then a position
    in a concatenated list of all columns is used.

    Parameters
    ----------
    idx : int
        Input column index.

    Attributes
    ----------
    input : int
        Input column index.
    """

    def __init__(self, idx):
        self.input = idx

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        CalciteInputRefExpr
        """
        return CalciteInputRefExpr(self.input)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f'(input {self.input})'