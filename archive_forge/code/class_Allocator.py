from typing import TYPE_CHECKING, Tuple, Optional
import pyglet
class Allocator:
    """Rectangular area allocation algorithm.

    Initialise with a given ``width`` and ``height``, then repeatedly
    call `alloc` to retrieve free regions of the area and protect that
    area from future allocations.

    `Allocator` uses a fairly simple strips-based algorithm.  It performs best
    when rectangles are allocated in decreasing height order.
    """
    __slots__ = ('width', 'height', 'strips', 'used_area')

    def __init__(self, width: int, height: int) -> None:
        """Create an `Allocator` of the given size.

        :Parameters:
            `width` : int
                Width of the allocation region.
            `height` : int
                Height of the allocation region.

        """
        assert width > 0 and height > 0
        self.width = width
        self.height = height
        self.strips = [_Strip(0, height)]
        self.used_area = 0

    def alloc(self, width: int, height: int) -> Tuple[int, int]:
        """Get a free area in the allocator of the given size.

        After calling `alloc`, the requested area will no longer be used.
        If there is not enough room to fit the given area `AllocatorException`
        is raised.

        :Parameters:
            `width` : int
                Width of the area to allocate.
            `height` : int
                Height of the area to allocate.

        :rtype: int, int
        :return: The X and Y coordinates of the bottom-left corner of the
            allocated region.
        """
        for strip in self.strips:
            if self.width - strip.x >= width and strip.max_height >= height:
                self.used_area += width * height
                return strip.add(width, height)
        if self.width >= width and self.height - strip.y2 >= height:
            self.used_area += width * height
            strip.compact()
            newstrip = _Strip(strip.y2, self.height - strip.y2)
            self.strips.append(newstrip)
            return newstrip.add(width, height)
        raise AllocatorException('No more space in %r for box %dx%d' % (self, width, height))

    def get_usage(self) -> float:
        """Get the fraction of area already allocated.

        This method is useful for debugging and profiling only.

        :rtype: float
        """
        return self.used_area / float(self.width * self.height)

    def get_fragmentation(self) -> float:
        """Get the fraction of area that's unlikely to ever be used, based on
        current allocation behaviour.

        This method is useful for debugging and profiling only.

        :rtype: float
        """
        if not self.strips:
            return 0.0
        possible_area = self.strips[-1].y2 * self.width
        return 1.0 - self.used_area / possible_area