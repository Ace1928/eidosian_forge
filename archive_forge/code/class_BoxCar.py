import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class BoxCar(NumberGenerator, TimeDependent):
    """
    The boxcar function over the specified time interval. The bounds
    are exclusive: zero is returned at the onset time and at the
    offset (onset+duration).

    If duration is None, then this reduces to a step function around the
    onset value with no offset.

    See http://en.wikipedia.org/wiki/Boxcar_function
    """
    onset = param.Number(0.0, doc='Time of onset.')
    duration = param.Number(None, allow_None=True, bounds=(0.0, None), doc='\n        Duration of step value.')

    def __call__(self):
        if self.time_fn() <= self.onset:
            return 0.0
        elif self.duration is not None and self.time_fn() > self.onset + self.duration:
            return 0.0
        else:
            return 1.0