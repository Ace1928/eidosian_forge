import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def _calcWidths(self):
    """Vector of widths for stringWidth function"""
    w = [0] * 256
    gw = self.face.glyphWidths
    vec = self.encoding.vector
    for i in range(256):
        glyphName = vec[i]
        if glyphName is not None:
            try:
                width = gw[glyphName]
                w[i] = width
            except KeyError:
                import reportlab.rl_config
                if reportlab.rl_config.warnOnMissingFontGlyphs:
                    print('typeface "%s" does not have a glyph "%s", bad font!' % (self.face.name, glyphName))
                else:
                    pass
    self.widths = w