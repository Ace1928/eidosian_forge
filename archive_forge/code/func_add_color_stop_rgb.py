from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def add_color_stop_rgb(self, offset, red, green, blue):
    """Same as :meth:`add_color_stop_rgba` with ``alpha=1``.
        Kept for compatibility with pycairo.

        """
    cairo.cairo_pattern_add_color_stop_rgb(self._pointer, offset, red, green, blue)
    self._check_status()