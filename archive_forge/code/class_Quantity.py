from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
class Quantity:

    def __init__(self, data, units):
        self.magnitude = data
        self.units = units

    def to(self, new_units):
        factors = {('hours', 'seconds'): 3600, ('minutes', 'hours'): 1 / 60, ('minutes', 'seconds'): 60, ('feet', 'miles'): 1 / 5280.0, ('feet', 'inches'): 12, ('miles', 'inches'): 12 * 5280}
        if self.units != new_units:
            mult = factors[self.units, new_units]
            return Quantity(mult * self.magnitude, new_units)
        else:
            return Quantity(self.magnitude, self.units)

    def __copy__(self):
        return Quantity(self.magnitude, self.units)

    def __getattr__(self, attr):
        return getattr(self.magnitude, attr)

    def __getitem__(self, item):
        if np.iterable(self.magnitude):
            return Quantity(self.magnitude[item], self.units)
        else:
            return Quantity(self.magnitude, self.units)

    def __array__(self):
        return np.asarray(self.magnitude)