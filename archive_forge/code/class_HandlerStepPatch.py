from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerStepPatch(HandlerBase):
    """
    Handler for `~.matplotlib.patches.StepPatch` instances.
    """

    @staticmethod
    def _create_patch(orig_handle, xdescent, ydescent, width, height):
        return Rectangle(xy=(-xdescent, -ydescent), width=width, height=height, color=orig_handle.get_facecolor())

    @staticmethod
    def _create_line(orig_handle, width, height):
        legline = Line2D([0, width], [height / 2, height / 2], color=orig_handle.get_edgecolor(), linestyle=orig_handle.get_linestyle(), linewidth=orig_handle.get_linewidth())
        legline.set_drawstyle('default')
        legline.set_marker('')
        return legline

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        if orig_handle.get_fill() or orig_handle.get_hatch() is not None:
            p = self._create_patch(orig_handle, xdescent, ydescent, width, height)
            self.update_prop(p, orig_handle, legend)
        else:
            p = self._create_line(orig_handle, width, height)
        p.set_transform(trans)
        return [p]