import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class NullLocator(Locator):
    """
    No ticks
    """

    def __call__(self):
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are Null, vmin and vmax are not used in this
            method.
        """
        return []