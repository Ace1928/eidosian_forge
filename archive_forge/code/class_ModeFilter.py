from __future__ import annotations
import functools
class ModeFilter(Filter):
    """
    Create a mode filter. Picks the most frequent pixel value in a box with the
    given size.  Pixel values that occur only once or twice are ignored; if no
    pixel value occurs more than twice, the original pixel value is preserved.

    :param size: The kernel size, in pixels.
    """
    name = 'Mode'

    def __init__(self, size=3):
        self.size = size

    def filter(self, image):
        return image.modefilter(self.size)