from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerRegularPolyCollection(HandlerNpointsYoffsets):
    """Handler for `.RegularPolyCollection`\\s."""

    def __init__(self, yoffsets=None, sizes=None, **kwargs):
        super().__init__(yoffsets=yoffsets, **kwargs)
        self._sizes = sizes

    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints

    def get_sizes(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        if self._sizes is None:
            handle_sizes = orig_handle.get_sizes()
            if not len(handle_sizes):
                handle_sizes = [1]
            size_max = max(handle_sizes) * legend.markerscale ** 2
            size_min = min(handle_sizes) * legend.markerscale ** 2
            numpoints = self.get_numpoints(legend)
            if numpoints < 4:
                sizes = [0.5 * (size_max + size_min), size_max, size_min][:numpoints]
            else:
                rng = size_max - size_min
                sizes = rng * np.linspace(0, 1, numpoints) + size_min
        else:
            sizes = self._sizes
        return sizes

    def update_prop(self, legend_handle, orig_handle, legend):
        self._update_prop(legend_handle, orig_handle)
        legend_handle.set_figure(legend.figure)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        return type(orig_handle)(orig_handle.get_numsides(), rotation=orig_handle.get_rotation(), sizes=sizes, offsets=offsets, offset_transform=offset_transform)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        ydata = self.get_ydata(legend, xdescent, ydescent, width, height, fontsize)
        sizes = self.get_sizes(legend, orig_handle, xdescent, ydescent, width, height, fontsize)
        p = self.create_collection(orig_handle, sizes, offsets=list(zip(xdata_marker, ydata)), offset_transform=trans)
        self.update_prop(p, orig_handle, legend)
        p.set_offset_transform(trans)
        return [p]