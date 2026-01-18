import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def _formatWidths(self):
    """returns a pretty block in PDF Array format to aid inspection"""
    text = b'['
    for i in range(256):
        text = text + b' ' + bytes(str(self.widths[i]), 'utf8')
        if i == 255:
            text = text + b' ]'
        if i % 16 == 15:
            text = text + b'\n'
    return text