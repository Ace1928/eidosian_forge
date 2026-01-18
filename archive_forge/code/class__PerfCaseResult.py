import math as _math
import time as _time
import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.api import runtime
class _PerfCaseResult:
    """ An obscure object encompassing timing results recorded by
    :func:`~cupyx.profiler.benchmark`. Simple statistics can be obtained by
    converting an instance of this class to a string.

    .. warning::
        This API is currently experimental and subject to change in future
        releases.

    """

    def __init__(self, name, ts, devices):
        assert ts.ndim == 2
        assert ts.shape[0] == len(devices) + 1
        assert ts.shape[1] > 0
        self.name = name
        self._ts = ts
        self._devices = devices

    def __repr__(self) -> str:
        """ Returns a string representation of the object.

        Returns:
            str: A string representation of the object.
        """
        return self.to_str(show_gpu=True)

    @property
    def cpu_times(self) -> _numpy.ndarray:
        """A :class:`numpy.ndarray` of shape ``(n_repeat,)``, holding times spent
        on CPU in seconds.

        These values are delta of the host-side performance counter
        (:func:`time.perf_counter`) between each repeat step.
        """
        return self._ts[0]

    @property
    def gpu_times(self) -> _numpy.ndarray:
        """A :class:`numpy.ndarray` of shape ``(len(devices), n_repeat)``,
        holding times spent on GPU in seconds.

        These values are measured using ``cudaEventElapsedTime`` with events
        recoreded before/after each repeat step.
        """
        return self._ts[1:]

    @staticmethod
    def _to_str_per_item(device_name, t):
        assert t.ndim == 1
        assert t.size > 0
        t_us = t * 1000000.0
        s = '    {}: {:9.03f} us'.format(device_name, t_us.mean())
        if t.size > 1:
            s += '   +/- {:6.03f} (min: {:9.03f} / max: {:9.03f}) us'.format(t_us.std(), t_us.min(), t_us.max())
        return s

    def to_str(self, show_gpu=False):
        results = [self._to_str_per_item('CPU', self._ts[0])]
        if show_gpu:
            for i, d in enumerate(self._devices):
                results.append(self._to_str_per_item('GPU-{}'.format(d), self._ts[1 + i]))
        return '{:<20s}:{}'.format(self.name, ' '.join(results))

    def __str__(self):
        return self.to_str(show_gpu=True)