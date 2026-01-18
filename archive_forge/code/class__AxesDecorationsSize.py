from numbers import Real
from matplotlib import _api
from matplotlib.axes import Axes
class _AxesDecorationsSize(_Base):
    """
    Fixed size, corresponding to the size of decorations on a given Axes side.
    """
    _get_size_map = {'left': lambda tight_bb, axes_bb: axes_bb.xmin - tight_bb.xmin, 'right': lambda tight_bb, axes_bb: tight_bb.xmax - axes_bb.xmax, 'bottom': lambda tight_bb, axes_bb: axes_bb.ymin - tight_bb.ymin, 'top': lambda tight_bb, axes_bb: tight_bb.ymax - axes_bb.ymax}

    def __init__(self, ax, direction):
        self._get_size = _api.check_getitem(self._get_size_map, direction=direction)
        self._ax_list = [ax] if isinstance(ax, Axes) else ax

    def get_size(self, renderer):
        sz = max([self._get_size(ax.get_tightbbox(renderer, call_axes_locator=False), ax.bbox) for ax in self._ax_list])
        dpi = renderer.points_to_pixels(72)
        abs_size = sz / dpi
        rel_size = 0
        return (rel_size, abs_size)