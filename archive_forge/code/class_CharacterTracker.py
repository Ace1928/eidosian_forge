from io import BytesIO
import functools
from fontTools import subset
import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase
class CharacterTracker:
    """
    Helper for font subsetting by the pdf and ps backends.

    Maintains a mapping of font paths to the set of character codepoints that
    are being used from that font.
    """

    def __init__(self):
        self.used = {}

    def track(self, font, s):
        """Record that string *s* is being typeset using font *font*."""
        char_to_font = font._get_fontmap(s)
        for _c, _f in char_to_font.items():
            self.used.setdefault(_f.fname, set()).add(ord(_c))

    def track_glyph(self, font, glyph):
        """Record that codepoint *glyph* is being typeset using font *font*."""
        self.used.setdefault(font.fname, set()).add(glyph)