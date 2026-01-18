import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def _pfbCheck(p, d, m, fn):
    if chr(d[p]) != PFB_MARKER or chr(d[p + 1]) != m:
        raise ValueError("Bad pfb file'%s' expected chr(%d)chr(%d) at char %d, got chr(%d)chr(%d)" % (fn, ord(PFB_MARKER), ord(m), p, d[p], d[p + 1]))
    if m == PFB_EOF:
        return
    p = p + 2
    l = (d[p + 3] << 8 | d[p + 2] << 8 | d[p + 1]) << 8 | d[p]
    p = p + 4
    if p + l > len(d):
        raise ValueError("Bad pfb file'%s' needed %d+%d bytes have only %d!" % (fn, p, l, len(d)))
    return (p, p + l)