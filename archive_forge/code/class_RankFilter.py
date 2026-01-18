from __future__ import annotations
import functools
class RankFilter(Filter):
    """
    Create a rank filter.  The rank filter sorts all pixels in
    a window of the given size, and returns the ``rank``'th value.

    :param size: The kernel size, in pixels.
    :param rank: What pixel value to pick.  Use 0 for a min filter,
                 ``size * size / 2`` for a median filter, ``size * size - 1``
                 for a max filter, etc.
    """
    name = 'Rank'

    def __init__(self, size, rank):
        self.size = size
        self.rank = rank

    def filter(self, image):
        if image.mode == 'P':
            msg = 'cannot filter palette images'
            raise ValueError(msg)
        image = image.expand(self.size // 2, self.size // 2)
        return image.rankfilter(self.size, self.rank)