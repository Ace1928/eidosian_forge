import functools
import operator
import math
import datetime as DT
from matplotlib import _api
from matplotlib.dates import date2num

        Generate a range of Epoch objects.

        Similar to the Python range() method.  Returns the range [
        start, stop) at the requested step.  Each element will be a
        Epoch object.

        = INPUT VARIABLES
        - start     The starting value of the range.
        - stop      The stop value of the range.
        - step      Step to use.

        = RETURN VALUE
        - Returns a list containing the requested Epoch values.
        