from numbers import Real
from matplotlib import _api
from matplotlib.axes import Axes
class MaxHeight(MaxExtent):
    """
    Size whose absolute part is the largest height of the given *artist_list*.
    """

    def __init__(self, artist_list):
        super().__init__(artist_list, 'height')