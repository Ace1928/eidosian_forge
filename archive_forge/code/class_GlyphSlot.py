import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class GlyphSlot(object):
    """
    FT_GlyphSlot wrapper.

    FreeType root glyph slot class structure. A glyph slot is a container where
    individual glyphs can be loaded, be they in outline or bitmap format.
    """

    def __init__(self, slot):
        """
        Create GlyphSlot object from an FT glyph slot.

        Parameters:
        -----------
          glyph: valid FT_GlyphSlot object
        """
        self._FT_GlyphSlot = slot

    def render(self, render_mode):
        """
        Convert a given glyph image to a bitmap. It does so by inspecting the
        glyph image format, finding the relevant renderer, and invoking it.
        
        :param render_mode: The render mode used to render the glyph image into
                            a bitmap. See FT_Render_Mode for a list of possible
                            values.
                            
                            If FT_RENDER_MODE_NORMAL is used, a previous call
                            of FT_Load_Glyph with flag FT_LOAD_COLOR makes
                            FT_Render_Glyph provide a default blending of
                            colored glyph layers associated with the current
                            glyph slot (provided the font contains such layers)
                            instead of rendering the glyph slot's outline.
                            This is an experimental feature; see FT_LOAD_COLOR
                            for more information.
                            
        **Note**:
        
          To get meaningful results, font scaling values must be set with
          functions like FT_Set_Char_Size before calling FT_Render_Glyph.

          When FreeType outputs a bitmap of a glyph, it really outputs an alpha
          coverage map. If a pixel is completely covered by a filled-in
          outline, the bitmap contains 0xFF at that pixel, meaning that
          0xFF/0xFF fraction of that pixel is covered, meaning the pixel is
          100% black (or 0% bright). If a pixel is only 50% covered
          (value 0x80), the pixel is made 50% black (50% bright or a middle
          shade of grey). 0% covered means 0% black (100% bright or white).

          On high-DPI screens like on smartphones and tablets, the pixels are
          so small that their chance of being completely covered and therefore
          completely black are fairly good. On the low-DPI screens, however,
          the situation is different. The pixels are too large for most of the
          details of a glyph and shades of gray are the norm rather than the
          exception.

          This is relevant because all our screens have a second problem: they
          are not linear. 1 + 1 is not 2. Twice the value does not result in
          twice the brightness. When a pixel is only 50% covered, the coverage
          map says 50% black, and this translates to a pixel value of 128 when
          you use 8 bits per channel (0-255). However, this does not translate
          to 50% brightness for that pixel on our sRGB and gamma 2.2 screens.
          Due to their non-linearity, they dwell longer in the darks and only a
          pixel value of about 186 results in 50% brightness – 128 ends up too
          dark on both bright and dark backgrounds. The net result is that dark
          text looks burnt-out, pixely and blotchy on bright background, bright
          text too frail on dark backgrounds, and colored text on colored
          background (for example, red on green) seems to have dark halos or
          ‘dirt’ around it. The situation is especially ugly for diagonal stems
          like in ‘w’ glyph shapes where the quality of FreeType's
          anti-aliasing depends on the correct display of grays. On high-DPI
          screens where smaller, fully black pixels reign supreme, this doesn't
          matter, but on our low-DPI screens with all the gray shades, it does.
          0% and 100% brightness are the same things in linear and non-linear
          space, just all the shades in-between aren't.

          The blending function for placing text over a background is

          dst = alpha * src + (1 - alpha) * dst

          which is known as the OVER operator.

          To correctly composite an anti-aliased pixel of a glyph onto a
          surface, take the foreground and background colors (e.g., in sRGB
          space) and apply gamma to get them in a linear space, use OVER to
          blend the two linear colors using the glyph pixel as the alpha value
          (remember, the glyph bitmap is an alpha coverage bitmap), and apply
          inverse gamma to the blended pixel and write it back to the image.

          Internal testing at Adobe found that a target inverse gamma of 1.8
          for step 3 gives good results across a wide range of displays with
          an sRGB gamma curve or a similar one.

          This process can cost performance. There is an approximation that
          does not need to know about the background color; see
          https://bel.fi/alankila/lcd/ and
          https://bel.fi/alankila/lcd/alpcor.html for details.

          **ATTENTION:** Linear blending is even more important when dealing
          with subpixel-rendered glyphs to prevent color-fringing! A
          subpixel-rendered glyph must first be filtered with a filter that
          gives equal weight to the three color primaries and does not exceed a
          sum of 0x100, see section ‘Subpixel Rendering’. Then the only
          difference to gray linear blending is that subpixel-rendered linear
          blending is done 3 times per pixel: red foreground subpixel to red
          background subpixel and so on for green and blue.
        """
        error = FT_Render_Glyph(self._FT_GlyphSlot, render_mode)
        if error:
            raise FT_Exception(error)

    def get_glyph(self):
        """
        A function used to extract a glyph image from a slot. Note that the
        created FT_Glyph object must be released with FT_Done_Glyph.
        """
        aglyph = FT_Glyph()
        error = FT_Get_Glyph(self._FT_GlyphSlot, byref(aglyph))
        if error:
            raise FT_Exception(error)
        return Glyph(aglyph)

    def _get_bitmap(self):
        return Bitmap(self._FT_GlyphSlot.contents.bitmap)
    bitmap = property(_get_bitmap, doc='This field is used as a bitmap descriptor when the slot format\n                is FT_GLYPH_FORMAT_BITMAP. Note that the address and content of\n                the bitmap buffer can change between calls of FT_Load_Glyph and\n                a few other functions.')

    def _get_metrics(self):
        return GlyphMetrics(self._FT_GlyphSlot.contents.metrics)
    metrics = property(_get_metrics, doc='The metrics of the last loaded glyph in the slot. The returned\n       values depend on the last load flags (see the FT_Load_Glyph API\n       function) and can be expressed either in 26.6 fractional pixels or font\n       units. Note that even when the glyph image is transformed, the metrics\n       are not.')

    def _get_next(self):
        return GlyphSlot(self._FT_GlyphSlot.contents.next)
    next = property(_get_next, doc="In some cases (like some font tools), several glyph slots per\n              face object can be a good thing. As this is rare, the glyph slots\n              are listed through a direct, single-linked list using its 'next'\n              field.")
    advance = property(lambda self: self._FT_GlyphSlot.contents.advance, doc="This shorthand is, depending on FT_LOAD_IGNORE_TRANSFORM, the\n                 transformed advance width for the glyph (in 26.6 fractional\n                 pixel format). As specified with FT_LOAD_VERTICAL_LAYOUT, it\n                 uses either the 'horiAdvance' or the 'vertAdvance' value of\n                 'metrics' field.")

    def _get_outline(self):
        return Outline(self._FT_GlyphSlot.contents.outline)
    outline = property(_get_outline, doc="The outline descriptor for the current glyph image if its\n                 format is FT_GLYPH_FORMAT_OUTLINE. Once a glyph is loaded,\n                 'outline' can be transformed, distorted, embolded,\n                 etc. However, it must not be freed.")
    format = property(lambda self: self._FT_GlyphSlot.contents.format, doc='This field indicates the format of the image contained in the\n                glyph slot. Typically FT_GLYPH_FORMAT_BITMAP,\n                FT_GLYPH_FORMAT_OUTLINE, or FT_GLYPH_FORMAT_COMPOSITE, but\n                others are possible.')
    bitmap_top = property(lambda self: self._FT_GlyphSlot.contents.bitmap_top, doc="This is the bitmap's top bearing expressed in integer\n                     pixels. Remember that this is the distance from the\n                     baseline to the top-most glyph scanline, upwards y\n                     coordinates being positive.")
    bitmap_left = property(lambda self: self._FT_GlyphSlot.contents.bitmap_left, doc="This is the bitmap's left bearing expressed in integer\n                     pixels. Of course, this is only valid if the format is\n                     FT_GLYPH_FORMAT_BITMAP.")
    linearHoriAdvance = property(lambda self: self._FT_GlyphSlot.contents.linearHoriAdvance, doc='The advance width of the unhinted glyph. Its value\n                           is expressed in 16.16 fractional pixels, unless\n                           FT_LOAD_LINEAR_DESIGN is set when loading the glyph.\n                           This field can be important to perform correct\n                           WYSIWYG layout. Only relevant for outline glyphs.')
    linearVertAdvance = property(lambda self: self._FT_GlyphSlot.contents.linearVertAdvance, doc='The advance height of the unhinted glyph. Its value\n                           is expressed in 16.16 fractional pixels, unless\n                           FT_LOAD_LINEAR_DESIGN is set when loading the glyph.\n                           This field can be important to perform correct\n                           WYSIWYG layout. Only relevant for outline glyphs.')