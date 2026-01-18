from collections import OrderedDict
import dateutil.parser
import itertools
import logging
import numpy as np
from matplotlib import _api, ticker, units
class StrCategoryFormatter(ticker.Formatter):
    """String representation of the data at every tick."""

    def __init__(self, units_mapping):
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        self._units = units_mapping

    def __call__(self, x, pos=None):
        return self.format_ticks([x])[0]

    def format_ticks(self, values):
        r_mapping = {v: self._text(k) for k, v in self._units.items()}
        return [r_mapping.get(round(val), '') for val in values]

    @staticmethod
    def _text(value):
        """Convert text values into utf-8 or ascii strings."""
        if isinstance(value, bytes):
            value = value.decode(encoding='utf-8')
        elif not isinstance(value, str):
            value = str(value)
        return value