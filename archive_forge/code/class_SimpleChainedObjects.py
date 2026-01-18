import matplotlib.axes as maxes
from matplotlib.artist import Artist
from matplotlib.axis import XAxis, YAxis
class SimpleChainedObjects:

    def __init__(self, objects):
        self._objects = objects

    def __getattr__(self, k):
        _a = SimpleChainedObjects([getattr(a, k) for a in self._objects])
        return _a

    def __call__(self, *args, **kwargs):
        for m in self._objects:
            m(*args, **kwargs)