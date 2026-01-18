from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerLine2DCompound(HandlerNpoints):
    """
    Original handler for `.Line2D` instances, that relies on combining
    a line-only with a marker-only artist.  May be deprecated in the future.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata)
        self.update_prop(legline, orig_handle, legend)
        legline.set_drawstyle('default')
        legline.set_marker('')
        legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        self.update_prop(legline_marker, orig_handle, legend)
        legline_marker.set_linestyle('None')
        if legend.markerscale != 1:
            newsz = legline_marker.get_markersize() * legend.markerscale
            legline_marker.set_markersize(newsz)
        legline._legmarker = legline_marker
        legline.set_transform(trans)
        legline_marker.set_transform(trans)
        return [legline, legline_marker]