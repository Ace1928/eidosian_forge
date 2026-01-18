from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
class FontFace(object):
    """The base class for all font face types.

    Should not be instantiated directly, but see :doc:`cffi_api`.
    An instance may be returned for cairo font face types
    that are not (yet) defined in cairocffi.

    """

    def __init__(self, pointer):
        self._pointer = ffi.gc(pointer, _keepref(cairo, cairo.cairo_font_face_destroy))
        self._check_status()

    def _check_status(self):
        _check_status(cairo.cairo_font_face_status(self._pointer))

    @staticmethod
    def _from_pointer(pointer, incref):
        """Wrap an existing ``cairo_font_face_t *`` cdata pointer.

        :type incref: bool
        :param incref:
            Whether increase the :ref:`reference count <refcounting>` now.
        :return:
            A new instance of :class:`FontFace` or one of its sub-classes,
            depending on the face’s type.

        """
        if pointer == ffi.NULL:
            raise ValueError('Null pointer')
        if incref:
            cairo.cairo_font_face_reference(pointer)
        self = object.__new__(FONT_TYPE_TO_CLASS.get(cairo.cairo_font_face_get_type(pointer), FontFace))
        FontFace.__init__(self, pointer)
        return self