from param.parameterized import get_occupied_slots
from .util import datetime_types
class BoundingRegion:
    """
    Abstract bounding region class, for any portion of a 2D plane.

    Only subclasses can be instantiated directly.
    """
    __abstract = True
    __slots__ = ['_aarect']

    def contains(self, x, y):
        raise NotImplementedError

    def __contains__(self, point):
        x, y = point
        return self.contains(x, y)

    def scale(self, xs, ys):
        raise NotImplementedError

    def translate(self, xoff, yoff):
        l, b, r, t = self.aarect().lbrt()
        self._aarect = AARectangle((l + xoff, b + yoff), (r + xoff, t + yoff))

    def rotate(self, theta):
        raise NotImplementedError

    def aarect(self):
        raise NotImplementedError

    def centroid(self):
        """
        Return the coordinates of the center of this BoundingBox
        """
        return self.aarect().centroid()

    def set(self, points):
        self._aarect = AARectangle(*points)

    def __getstate__(self):
        state = {}
        for slot in get_occupied_slots(self):
            state[slot] = getattr(self, slot)
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)