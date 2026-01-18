from numbers import Real
from matplotlib import _api
from matplotlib.axes import Axes
class MaxExtent(_Base):
    """
    Size whose absolute part is either the largest width or the largest height
    of the given *artist_list*.
    """

    def __init__(self, artist_list, w_or_h):
        self._artist_list = artist_list
        _api.check_in_list(['width', 'height'], w_or_h=w_or_h)
        self._w_or_h = w_or_h

    def add_artist(self, a):
        self._artist_list.append(a)

    def get_size(self, renderer):
        rel_size = 0.0
        extent_list = [getattr(a.get_window_extent(renderer), self._w_or_h) / a.figure.dpi for a in self._artist_list]
        abs_size = max(extent_list, default=0)
        return (rel_size, abs_size)