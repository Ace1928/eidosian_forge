from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
class ScaledFont(object):
    """Creates a :class:`ScaledFont` object from a font face and matrices
    that describe the size of the font
    and the environment in which it will be used.

    :param font_face: A :class:`FontFace` object.
    :type font_matrix: Matrix
    :param font_matrix:
        Font space to user space transformation matrix for the font.
        In the simplest case of a N point font,
        this matrix is just a scale by N,
        but it can also be used to shear the font
        or stretch it unequally along the two axes.
        If omitted, a scale by 10 matrix is assumed (ie. a 10 point font size).
        See :class:`Context.set_font_matrix`.
    :type ctm: Matrix
    :param ctm:
        User to device transformation matrix with which the font will be used.
        If omitted, an identity matrix is assumed.
    :param options:
        The :class:`FontOptions` object to use
        when getting metrics for the font and rendering with it.
        If omitted, the default options are assumed.

    """

    def __init__(self, font_face, font_matrix=None, ctm=None, options=None):
        if font_matrix is None:
            font_matrix = Matrix()
            font_matrix.scale(10)
        if ctm is None:
            ctm = Matrix()
        if options is None:
            options = FontOptions()
        self._init_pointer(cairo.cairo_scaled_font_create(font_face._pointer, font_matrix._pointer, ctm._pointer, options._pointer))

    def _init_pointer(self, pointer):
        self._pointer = ffi.gc(pointer, _keepref(cairo, cairo.cairo_scaled_font_destroy))
        self._check_status()

    def _check_status(self):
        _check_status(cairo.cairo_scaled_font_status(self._pointer))

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

    def get_font_face(self):
        """Return the font face that this scaled font uses.

        :returns:
            A new instance of :class:`FontFace` (or one of its sub-classes).
            Might wrap be the same font face passed to :class:`ScaledFont`,
            but this does not hold true for all possible cases.

        """
        return FontFace._from_pointer(cairo.cairo_scaled_font_get_font_face(self._pointer), incref=True)

    def get_font_options(self):
        """Copies the scaled font’s options.

        :returns: A new :class:`FontOptions` object.

        """
        font_options = FontOptions()
        cairo.cairo_scaled_font_get_font_options(self._pointer, font_options._pointer)
        return font_options

    def get_font_matrix(self):
        """Copies the scaled font’s font matrix.

        :returns: A new :class:`Matrix` object.

        """
        matrix = Matrix()
        cairo.cairo_scaled_font_get_font_matrix(self._pointer, matrix._pointer)
        self._check_status()
        return matrix

    def get_ctm(self):
        """Copies the scaled font’s font current transform matrix.

        Note that the translation offsets ``(x0, y0)`` of the CTM
        are ignored by :class:`ScaledFont`.
        So, the matrix this method returns always has 0 as ``x0`` and ``y0``.

        :returns: A new :class:`Matrix` object.

        """
        matrix = Matrix()
        cairo.cairo_scaled_font_get_ctm(self._pointer, matrix._pointer)
        self._check_status()
        return matrix

    def get_scale_matrix(self):
        """Copies the scaled font’s scaled matrix.

        The scale matrix is product of the font matrix
        and the ctm associated with the scaled font,
        and hence is the matrix mapping from font space to device space.

        :returns: A new :class:`Matrix` object.

        """
        matrix = Matrix()
        cairo.cairo_scaled_font_get_scale_matrix(self._pointer, matrix._pointer)
        self._check_status()
        return matrix

    def extents(self):
        """Return the scaled font’s extents.
        See :meth:`Context.font_extents`.

        :returns:
            A ``(ascent, descent, height, max_x_advance, max_y_advance)``
            tuple of floats.

        """
        extents = ffi.new('cairo_font_extents_t *')
        cairo.cairo_scaled_font_extents(self._pointer, extents)
        self._check_status()
        return (extents.ascent, extents.descent, extents.height, extents.max_x_advance, extents.max_y_advance)

    def text_extents(self, text):
        """Returns the extents for a string of text.

        The extents describe a user-space rectangle
        that encloses the "inked" portion of the text,
        (as it would be drawn by :meth:`Context.show_text`).
        Additionally, the ``x_advance`` and ``y_advance`` values
        indicate the amount by which the current point would be advanced
        by :meth:`Context.show_text`.

        :param text: The text to measure, as an Unicode or UTF-8 string.
        :returns:
            A ``(x_bearing, y_bearing, width, height, x_advance, y_advance)``
            tuple of floats.
            See :meth:`Context.text_extents` for details.

        """
        extents = ffi.new('cairo_text_extents_t *')
        cairo.cairo_scaled_font_text_extents(self._pointer, _encode_string(text), extents)
        self._check_status()
        return (extents.x_bearing, extents.y_bearing, extents.width, extents.height, extents.x_advance, extents.y_advance)

    def glyph_extents(self, glyphs):
        """Returns the extents for a list of glyphs.

        The extents describe a user-space rectangle
        that encloses the "inked" portion of the glyphs,
        (as it would be drawn by :meth:`Context.show_glyphs`).
        Additionally, the ``x_advance`` and ``y_advance`` values
        indicate the amount by which the current point would be advanced
        by :meth:`Context.show_glyphs`.

        :param glyphs:
            A list of glyphs, as returned by :meth:`text_to_glyphs`.
            Each glyph is a ``(glyph_id, x, y)`` tuple
            of an integer and two floats.
        :returns:
            A ``(x_bearing, y_bearing, width, height, x_advance, y_advance)``
            tuple of floats.
            See :meth:`Context.text_extents` for details.

        """
        glyphs = ffi.new('cairo_glyph_t[]', glyphs)
        extents = ffi.new('cairo_text_extents_t *')
        cairo.cairo_scaled_font_glyph_extents(self._pointer, glyphs, len(glyphs), extents)
        self._check_status()
        return (extents.x_bearing, extents.y_bearing, extents.width, extents.height, extents.x_advance, extents.y_advance)

    def text_to_glyphs(self, x, y, text, with_clusters):
        """Converts a string of text to a list of glyphs,
        optionally with cluster mapping,
        that can be used to render later using this scaled font.

        The output values can be readily passed to
        :meth:`Context.show_text_glyphs`, :meth:`Context.show_glyphs`
        or related methods,
        assuming that the exact same :class:`ScaledFont`
        is used for the operation.

        :type x: float
        :type y: float
        :type with_clusters: bool
        :param x: X position to place first glyph.
        :param y: Y position to place first glyph.
        :param text: The text to convert, as an Unicode or UTF-8 string.
        :param with_clusters: Whether to compute the cluster mapping.
        :returns:
            A ``(glyphs, clusters, clusters_flags)`` tuple
            if ``with_clusters`` is true, otherwise just ``glyphs``.
            See :meth:`Context.show_text_glyphs` for the data structure.

        .. note::

            This method is part of
            what the cairo designers call the "toy" text API.
            It is convenient for short demos and simple programs,
            but it is not expected to be adequate
            for serious text-using applications.
            See :ref:`fonts` for details
            and :meth:`Context.show_glyphs`
            for the "real" text display API in cairo.

        """
        glyphs = ffi.new('cairo_glyph_t **', ffi.NULL)
        num_glyphs = ffi.new('int *')
        if with_clusters:
            clusters = ffi.new('cairo_text_cluster_t **', ffi.NULL)
            num_clusters = ffi.new('int *')
            cluster_flags = ffi.new('cairo_text_cluster_flags_t *')
        else:
            clusters = ffi.NULL
            num_clusters = ffi.NULL
            cluster_flags = ffi.NULL
        status = cairo.cairo_scaled_font_text_to_glyphs(self._pointer, x, y, _encode_string(text), -1, glyphs, num_glyphs, clusters, num_clusters, cluster_flags)
        glyphs = ffi.gc(glyphs[0], _keepref(cairo, cairo.cairo_glyph_free))
        if with_clusters:
            clusters = ffi.gc(clusters[0], _keepref(cairo, cairo.cairo_text_cluster_free))
        _check_status(status)
        glyphs = [(glyph.index, glyph.x, glyph.y) for i in range(num_glyphs[0]) for glyph in [glyphs[i]]]
        if with_clusters:
            clusters = [(cluster.num_bytes, cluster.num_glyphs) for i in range(num_clusters[0]) for cluster in [clusters[i]]]
            return (glyphs, clusters, cluster_flags[0])
        else:
            return glyphs