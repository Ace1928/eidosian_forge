from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
class LinearGradient(Gradient):
    """Create a new linear gradient
    along the line defined by (x0, y0) and (x1, y1).
    Before using the gradient pattern, a number of color stops
    should be defined using :meth:`~Gradient.add_color_stop_rgba`.

    Note: The coordinates here are in pattern space.
    For a new pattern, pattern space is identical to user space,
    but the relationship between the spaces can be changed
    with :meth:`~Pattern.set_matrix`.

    :param x0: X coordinate of the start point.
    :param y0: Y coordinate of the start point.
    :param x1: X coordinate of the end point.
    :param y1: Y coordinate of the end point.
    :type x0: float
    :type y0: float
    :type x1: float
    :type y1: float

    """

    def __init__(self, x0, y0, x1, y1):
        Pattern.__init__(self, cairo.cairo_pattern_create_linear(x0, y0, x1, y1))

    def get_linear_points(self):
        """Return this linear gradient’s endpoints.

        :returns: A ``(x0, y0, x1, y1)`` tuple of floats.

        """
        points = ffi.new('double[4]')
        _check_status(cairo.cairo_pattern_get_linear_points(self._pointer, points + 0, points + 1, points + 2, points + 3))
        return tuple(points)