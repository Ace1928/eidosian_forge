from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerPathCollection(HandlerRegularPolyCollection):
    """Handler for `.PathCollection`\\s, which are used by `~.Axes.scatter`."""

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        return type(orig_handle)([orig_handle.get_paths()[0]], sizes=sizes, offsets=offsets, offset_transform=offset_transform)