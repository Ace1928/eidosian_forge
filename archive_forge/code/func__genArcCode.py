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
def _genArcCode(self, x1, y1, x2, y2, startAng, extent):
    """Calculate the path for an arc inscribed in rectangle defined by (x1,y1),(x2,y2)"""
    xScale = abs((x2 - x1) / 2.0)
    yScale = abs((y2 - y1) / 2.0)
    x, y = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    codeline = 'matrix currentmatrix %s %s translate %s %s scale 0 0 1 %s %s %s setmatrix'
    if extent >= 0:
        arc = 'arc'
    else:
        arc = 'arcn'
    data = (x, y, xScale, yScale, startAng, startAng + extent, arc)
    return codeline % data