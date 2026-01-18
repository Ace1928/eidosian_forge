from numbers import Real
from matplotlib import _api
from matplotlib.axes import Axes
class MaxWidth(MaxExtent):
    """
    Size whose absolute part is the largest width of the given *artist_list*.
    """

    def __init__(self, artist_list):
        super().__init__(artist_list, 'width')