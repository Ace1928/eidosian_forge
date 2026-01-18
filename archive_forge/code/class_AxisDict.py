import matplotlib.axes as maxes
from matplotlib.artist import Artist
from matplotlib.axis import XAxis, YAxis
class AxisDict(dict):

    def __init__(self, axes):
        self.axes = axes
        super().__init__()

    def __getitem__(self, k):
        if isinstance(k, tuple):
            r = SimpleChainedObjects([super(Axes.AxisDict, self).__getitem__(k1) for k1 in k])
            return r
        elif isinstance(k, slice):
            if k.start is None and k.stop is None and (k.step is None):
                return SimpleChainedObjects(list(self.values()))
            else:
                raise ValueError('Unsupported slice')
        else:
            return dict.__getitem__(self, k)

    def __call__(self, *v, **kwargs):
        return maxes.Axes.axis(self.axes, *v, **kwargs)