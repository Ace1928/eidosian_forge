import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def _loadBuiltInData(self, name):
    """Called for the built in 14 fonts.  Gets their glyph data.
        We presume they never change so this can be a shared reference."""
    name = str(name)
    self.glyphWidths = _fontdata.widthsByFontGlyph[name]
    self.glyphNames = list(self.glyphWidths.keys())
    self.ascent, self.descent = _fontdata.ascent_descent[name]