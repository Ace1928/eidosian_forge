import math
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import getFont, stringWidth, unicode2T1 # for font info
from reportlab.lib.utils import asBytes, char2int, rawBytes, asNative, isUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab import rl_config
from reportlab.pdfgen.canvas import FILL_EVEN_ODD
from reportlab.graphics.shapes import *
def _escape_and_limit(s):
    s = asBytes(s)
    R = []
    aR = R.append
    n = 0
    for c in s:
        c = _ESCAPEDICT[char2int(c)]
        aR(c)
        n += len(c)
        if n >= 200:
            n = 0
            aR('\\\n')
    return ''.join(R)