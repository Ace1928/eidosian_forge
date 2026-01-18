from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def _saveAsPICT(im, fn, fmt, transparent=None):
    im = _convert2pilp(im)
    cols, rows = im.size
    s = _pmBackend.pil2pict(cols, rows, (im.tobytes if hasattr(im, 'tobytes') else im.tostring)(), im.im.getpalette())
    if not hasattr(fn, 'write'):
        with open(os.path.splitext(fn)[0] + '.' + fmt.lower(), 'wb') as f:
            f.write(s)
        if os.name == 'mac':
            from reportlab.lib.utils import markfilename
            markfilename(fn, ext='PICT')
    else:
        fn.write(s)