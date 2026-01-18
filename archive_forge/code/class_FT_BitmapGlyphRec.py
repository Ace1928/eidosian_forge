from freetype.ft_types import *
class FT_BitmapGlyphRec(Structure):
    """
    A structure used for bitmap glyph images. This really is a 'sub-class' of
    FT_GlyphRec.
    """
    _fields_ = [('root', FT_GlyphRec), ('left', FT_Int), ('top', FT_Int), ('bitmap', FT_Bitmap)]