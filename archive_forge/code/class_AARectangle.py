from param.parameterized import get_occupied_slots
from .util import datetime_types
class AARectangle:
    """
    Axis-aligned rectangle class.

    Defines the smallest axis-aligned rectangle that encloses a set of
    points.

    Usage:  aar = AARectangle( (x1,y1),(x2,y2), ... , (xN,yN) )
    """
    __slots__ = ['_left', '_bottom', '_right', '_top']

    def __init__(self, *points):
        self._top = max([y for x, y in points])
        self._bottom = min([y for x, y in points])
        self._left = min([x for x, y in points])
        self._right = max([x for x, y in points])

    def __getstate__(self):
        state = {}
        for k in self.__slots__:
            state[k] = getattr(self, k)
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def top(self):
        """Return the y-coordinate of the top of the rectangle."""
        return self._top

    def bottom(self):
        """Return the y-coordinate of the bottom of the rectangle."""
        return self._bottom

    def left(self):
        """Return the x-coordinate of the left side of the rectangle."""
        return self._left

    def right(self):
        """Return the x-coordinate of the right side of the rectangle."""
        return self._right

    def lbrt(self):
        """Return (left,bottom,right,top) as a tuple."""
        return (self._left, self._bottom, self._right, self._top)

    def centroid(self):
        """
        Return the centroid of the rectangle.
        """
        left, bottom, right, top = self.lbrt()
        return ((right + left) / 2.0, (top + bottom) / 2.0)

    def intersect(self, other):
        l1, b1, r1, t1 = self.lbrt()
        l2, b2, r2, t2 = other.lbrt()
        l = max(l1, l2)
        b = max(b1, b2)
        r = min(r1, r2)
        t = min(t1, t2)
        return AARectangle(points=((l, b), (r, t)))

    def width(self):
        return self._right - self._left

    def height(self):
        return self._top - self._bottom

    def empty(self):
        l, b, r, t = self.lbrt()
        return r <= l or t <= b