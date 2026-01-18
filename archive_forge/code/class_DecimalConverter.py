from decimal import Decimal
from numbers import Number
import numpy as np
from numpy import ma
from matplotlib import cbook
class DecimalConverter(ConversionInterface):
    """Converter for decimal.Decimal data to float."""

    @staticmethod
    def convert(value, unit, axis):
        """
        Convert Decimals to floats.

        The *unit* and *axis* arguments are not used.

        Parameters
        ----------
        value : decimal.Decimal or iterable
            Decimal or list of Decimal need to be converted
        """
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, ma.MaskedArray):
            return ma.asarray(value, dtype=float)
        else:
            return np.asarray(value, dtype=float)