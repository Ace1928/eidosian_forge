import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class TimeSampledFn(NumberGenerator, TimeDependent):
    """
    Samples the values supplied by a time_dependent callable at
    regular intervals of duration 'period', with the sampled value
    held constant within each interval.
    """
    period = param.Number(default=1.0, bounds=(0.0, None), inclusive_bounds=(False, True), softbounds=(0.0, 5.0), doc='\n        The periodicity with which the values of fn are sampled.')
    offset = param.Number(default=0.0, bounds=(0.0, None), softbounds=(0.0, 5.0), doc='\n        The offset from time 0.0 at which the first sample will be drawn.\n        Must be less than the value of period.')
    fn = param.Callable(doc='\n        The time-dependent function used to generate the sampled values.')

    def __init__(self, **params):
        super().__init__(**params)
        if not getattr(self.fn, 'time_dependent', False):
            raise Exception("The function 'fn' needs to be time dependent.")
        if self.time_fn != self.fn.time_fn:
            raise Exception('Objects do not share the same time_fn')
        if self.offset >= self.period:
            raise Exception('The onset value must be less than the period.')

    def __call__(self):
        current_time = self.time_fn()
        current_time += self.offset
        difference = current_time % self.period
        with self.time_fn as t:
            t(current_time - difference - self.offset)
            value = self.fn()
        return value