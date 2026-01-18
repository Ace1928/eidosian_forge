from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
@staticmethod
def _from_pointer(pointer, incref):
    """Wrap an existing ``cairo_scaled_font_t *`` cdata pointer.

        :type incref: bool
        :param incref:
            Whether increase the :ref:`reference count <refcounting>` now.
        :return: A new :class:`ScaledFont` instance.

        """
    if pointer == ffi.NULL:
        raise ValueError('Null pointer')
    if incref:
        cairo.cairo_scaled_font_reference(pointer)
    self = object.__new__(ScaledFont)
    ScaledFont._init_pointer(self, pointer)
    return self