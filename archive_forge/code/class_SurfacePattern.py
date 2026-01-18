from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
class SurfacePattern(Pattern):
    """Create a new pattern for the given surface.

    :param surface: A :class:`Surface` object.

    """

    def __init__(self, surface):
        Pattern.__init__(self, cairo.cairo_pattern_create_for_surface(surface._pointer))

    def get_surface(self):
        """Return this :class:`SurfacePattern`’s surface.

        :returns:
            An instance of :class:`Surface` or one of its sub-classes,
            a new Python object referencing the existing cairo surface.

        """
        surface_p = ffi.new('cairo_surface_t **')
        _check_status(cairo.cairo_pattern_get_surface(self._pointer, surface_p))
        return Surface._from_pointer(surface_p[0], incref=True)