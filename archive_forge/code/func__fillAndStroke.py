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
def _fillAndStroke(self, code, clip=0, fill=1, stroke=1, fillMode=None):
    fill = self._fillColor and fill
    stroke = self._strokeColor and stroke
    if fill or stroke or clip:
        self.code.extend(code)
        if fill:
            if fillMode is None:
                fillMode = self._fillMode
            if stroke or clip:
                self.code_append('gsave')
            self.setColor(self._fillColor)
            self.code_append('eofill' if fillMode == FILL_EVEN_ODD else 'fill')
            if stroke or clip:
                self.code_append('grestore')
        if stroke:
            if clip:
                self.code_append('gsave')
            self.setColor(self._strokeColor)
            self.code_append('stroke')
            if clip:
                self.code_append('grestore')
        if clip:
            self.code_append('clip')
            self.code_append('newpath')