from __future__ import annotations
import functools
class MaxFilter(RankFilter):
    """
    Create a max filter.  Picks the largest pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """
    name = 'Max'

    def __init__(self, size=3):
        self.size = size
        self.rank = size * size - 1