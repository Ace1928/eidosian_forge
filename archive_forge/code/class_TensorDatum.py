from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class TensorDatum:
    """A single datum in a tensor time series for a run and tag.

    Attributes:
      step: The global step at which this datum occurred; an integer. This
        is a unique key among data of this time series.
      wall_time: The real-world time at which this datum occurred, as
        `float` seconds since epoch.
      numpy: The `numpy.ndarray` value with the tensor contents of this
        datum.
    """
    __slots__ = ('_step', '_wall_time', '_numpy')

    def __init__(self, step, wall_time, numpy):
        self._step = step
        self._wall_time = wall_time
        self._numpy = numpy

    @property
    def step(self):
        return self._step

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def numpy(self):
        return self._numpy

    def __eq__(self, other):
        if not isinstance(other, TensorDatum):
            return False
        if self._step != other._step:
            return False
        if self._wall_time != other._wall_time:
            return False
        if not np.array_equal(self._numpy, other._numpy):
            return False
        return True
    __hash__ = None

    def __repr__(self):
        return 'TensorDatum(%s)' % ', '.join(('step=%r' % (self._step,), 'wall_time=%r' % (self._wall_time,), 'numpy=%r' % (self._numpy,)))