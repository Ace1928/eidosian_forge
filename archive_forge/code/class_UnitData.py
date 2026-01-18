from collections import OrderedDict
import dateutil.parser
import itertools
import logging
import numpy as np
from matplotlib import _api, ticker, units
class UnitData:

    def __init__(self, data=None):
        """
        Create mapping between unique categorical values and integer ids.

        Parameters
        ----------
        data : iterable
            sequence of string values
        """
        self._mapping = OrderedDict()
        self._counter = itertools.count()
        if data is not None:
            self.update(data)

    @staticmethod
    def _str_is_convertible(val):
        """
        Helper method to check whether a string can be parsed as float or date.
        """
        try:
            float(val)
        except ValueError:
            try:
                dateutil.parser.parse(val)
            except (ValueError, TypeError):
                return False
        return True

    def update(self, data):
        """
        Map new values to integer identifiers.

        Parameters
        ----------
        data : iterable of str or bytes

        Raises
        ------
        TypeError
            If elements in *data* are neither str nor bytes.
        """
        data = np.atleast_1d(np.array(data, dtype=object))
        convertible = True
        for val in OrderedDict.fromkeys(data):
            _api.check_isinstance((str, bytes), value=val)
            if convertible:
                convertible = self._str_is_convertible(val)
            if val not in self._mapping:
                self._mapping[val] = next(self._counter)
        if data.size and convertible:
            _log.info('Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.')