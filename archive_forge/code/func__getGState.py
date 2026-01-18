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
@staticmethod
def _getGState(w, h, bg, backend=None, fmt='RGB24'):
    mod = _getPMBackend(backend)
    if backend is None:
        backend = rl_config.renderPMBackend
    if backend == '_renderPM':
        try:
            return mod.gstate(w, h, bg=bg)
        except TypeError:
            try:
                return mod.GState(w, h, bg, fmt=fmt)
            except:
                pass
    elif 'cairo' in backend.lower():
        try:
            return mod.GState(w, h, bg, fmt=fmt)
        except AttributeError:
            return mod.gstate(w, h, bg=bg)
    raise RuntimeError(f'Cannot obtain PM graphics state using backend {backend!r}')