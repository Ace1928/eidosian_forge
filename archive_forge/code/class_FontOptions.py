from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
class FontOptions(object):
    """An opaque object holding all options that are used when rendering fonts.

    Individual features of a :class:`FontOptions`
    can be set or accessed using method
    named ``set_FEATURE_NAME`` and ``get_FEATURE_NAME``,
    like :meth:`set_antialias` and :meth:`get_antialias`.

    New features may be added to :class:`FontOptions` in the future.
    For this reason, ``==``, :meth:`copy`, :meth:`merge`, and :func:`hash`
    should be used to check for equality copy,, merge,
    or compute a hash value of :class:`FontOptions` objects.

    :param values:
        Call the corresponding ``set_FEATURE_NAME`` methods
        after creating a new :class:`FontOptions`::

            options = FontOptions()
            options.set_antialias(cairocffi.ANTIALIAS_BEST)
            assert FontOptions(antialias=cairocffi.ANTIALIAS_BEST) == options

    """

    def __init__(self, **values):
        self._init_pointer(cairo.cairo_font_options_create())
        for name, value in values.items():
            getattr(self, 'set_' + name)(value)

    def _init_pointer(self, pointer):
        self._pointer = ffi.gc(pointer, _keepref(cairo, cairo.cairo_font_options_destroy))
        self._check_status()

    def _check_status(self):
        _check_status(cairo.cairo_font_options_status(self._pointer))

    def copy(self):
        """Return a new :class:`FontOptions` with the same values."""
        cls = type(self)
        other = object.__new__(cls)
        cls._init_pointer(other, cairo.cairo_font_options_copy(self._pointer))
        return other

    def merge(self, other):
        """Merges non-default options from ``other``,
        replacing existing values.
        This operation can be thought of as somewhat similar
        to compositing other onto options
        with the operation of :obj:`OVER <OPERATOR_OVER>`.

        """
        cairo.cairo_font_options_merge(self._pointer, other._pointer)
        _check_status(cairo.cairo_font_options_status(self._pointer))

    def __hash__(self):
        return cairo.cairo_font_options_hash(self._pointer)

    def __eq__(self, other):
        return cairo.cairo_font_options_equal(self._pointer, other._pointer)

    def __ne__(self, other):
        return not self == other
    equal = __eq__
    hash = __hash__

    def set_antialias(self, antialias):
        """Changes the :ref:`ANTIALIAS` for the font options object.
        This specifies the type of antialiasing to do when rendering text.

        """
        cairo.cairo_font_options_set_antialias(self._pointer, antialias)
        self._check_status()

    def get_antialias(self):
        """Return the :ref:`ANTIALIAS` string for the font options object."""
        return cairo.cairo_font_options_get_antialias(self._pointer)

    def set_subpixel_order(self, subpixel_order):
        """Changes the :ref:`SUBPIXEL_ORDER` for the font options object.
         The subpixel order specifies the order of color elements
         within each pixel on the display device
         when rendering with an antialiasing mode of
         :obj:`SUBPIXEL <ANTIALIAS_SUBPIXEL>`.

        """
        cairo.cairo_font_options_set_subpixel_order(self._pointer, subpixel_order)
        self._check_status()

    def get_subpixel_order(self):
        """Return the :ref:`SUBPIXEL_ORDER` string
        for the font options object.

        """
        return cairo.cairo_font_options_get_subpixel_order(self._pointer)

    def set_hint_style(self, hint_style):
        """Changes the :ref:`HINT_STYLE` for the font options object.
        This controls whether to fit font outlines to the pixel grid,
        and if so, whether to optimize for fidelity or contrast.

        """
        cairo.cairo_font_options_set_hint_style(self._pointer, hint_style)
        self._check_status()

    def get_hint_style(self):
        """Return the :ref:`HINT_STYLE` string for the font options object."""
        return cairo.cairo_font_options_get_hint_style(self._pointer)

    def set_hint_metrics(self, hint_metrics):
        """Changes the :ref:`HINT_METRICS` for the font options object.
        This controls whether metrics are quantized
        to integer values in device units.

        """
        cairo.cairo_font_options_set_hint_metrics(self._pointer, hint_metrics)
        self._check_status()

    def get_hint_metrics(self):
        """Return the :ref:`HINT_METRICS` string
        for the font options object.

        """
        return cairo.cairo_font_options_get_hint_metrics(self._pointer)

    def set_variations(self, variations):
        """Sets the OpenType font variations for the font options object.

        Font variations are specified as a string with a format that is similar
        to the CSS font-variation-settings. The string contains a
        comma-separated list of axis assignments, which each assignment
        consists of a 4-character axis name and a value, separated by
        whitespace and optional equals sign.

        :param variations: the new font variations, or ``None``.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
        if variations is None:
            variations = ffi.NULL
        else:
            variations = _encode_string(variations)
        cairo.cairo_font_options_set_variations(self._pointer, variations)
        self._check_status()

    def get_variations(self):
        """Gets the OpenType font variations for the font options object.

        See :meth:`set_variations` for details about the
        string format.

        :return: the font variations for the font options object. The
          returned string belongs to the ``options`` and must not be modified.
          It is valid until either the font options object is destroyed or the
          font variations in this object is modified with
          :meth:`set_variations`.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
        variations = cairo.cairo_font_options_get_variations(self._pointer)
        if variations != ffi.NULL:
            return ffi.string(variations).decode('utf8', 'replace')