import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteInputIdxExpr(BaseExpr):
    """
    Basically the same as ``CalciteInputRefExpr`` but with a different serialization.

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
        CalciteInputIdxExpr
        """
        return CalciteInputIdxExpr(self.input)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f'(input_idx {self.input})'