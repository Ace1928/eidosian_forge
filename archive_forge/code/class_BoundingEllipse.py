from param.parameterized import get_occupied_slots
from .util import datetime_types
class BoundingEllipse(BoundingBox):
    """
    Similar to BoundingBox, but the region is the ellipse
    inscribed within the rectangle.
    """
    __slots__ = []

    def contains(self, x, y):
        left, bottom, right, top = self.aarect().lbrt()
        xr = (right - left) / 2.0
        yr = (top - bottom) / 2.0
        xc = left + xr
        yc = bottom + yr
        xd = x - xc
        yd = y - yc
        return xd ** 2 / xr ** 2 + yd ** 2 / yr ** 2 <= 1