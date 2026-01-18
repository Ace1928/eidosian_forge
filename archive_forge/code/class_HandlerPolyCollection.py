from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerPolyCollection(HandlerBase):
    """
    Handler for `.PolyCollection` used in `~.Axes.fill_between` and
    `~.Axes.stackplot`.
    """

    def _update_prop(self, legend_handle, orig_handle):

        def first_color(colors):
            if colors.size == 0:
                return (0, 0, 0, 0)
            return tuple(colors[0])

        def get_first(prop_array):
            if len(prop_array):
                return prop_array[0]
            else:
                return None
        legend_handle._facecolor = first_color(orig_handle.get_facecolor())
        legend_handle._edgecolor = first_color(orig_handle.get_edgecolor())
        legend_handle._original_facecolor = orig_handle._original_facecolor
        legend_handle._original_edgecolor = orig_handle._original_edgecolor
        legend_handle._fill = orig_handle.get_fill()
        legend_handle._hatch = orig_handle.get_hatch()
        legend_handle._hatch_color = orig_handle._hatch_color
        legend_handle.set_linewidth(get_first(orig_handle.get_linewidths()))
        legend_handle.set_linestyle(get_first(orig_handle.get_linestyles()))
        legend_handle.set_transform(get_first(orig_handle.get_transforms()))
        legend_handle.set_figure(orig_handle.get_figure())

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = Rectangle(xy=(-xdescent, -ydescent), width=width, height=height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]