import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
class BulletDrawer:

    def __init__(self, value='0', bulletAlign='left', bulletType='1', bulletColor='black', bulletFontName='Helvetica', bulletFontSize=12, bulletOffsetY=0, bulletDedent=0, bulletDir='ltr', bulletFormat=None):
        self.value = value
        self._bulletAlign = bulletAlign
        self._bulletType = bulletType
        self._bulletColor = bulletColor
        self._bulletFontName = bulletFontName
        self._bulletFontSize = bulletFontSize
        self._bulletOffsetY = bulletOffsetY
        self._bulletDedent = bulletDedent
        self._bulletDir = bulletDir
        self._bulletFormat = bulletFormat

    def drawOn(self, indenter, canv, x, y, _sW=0):
        value = self.value
        if not value:
            return
        canv.saveState()
        canv.translate(x, y)
        y = indenter.height - self._bulletFontSize + self._bulletOffsetY
        if self._bulletDir == 'rtl':
            x = indenter.width - indenter._rightIndent + self._bulletDedent
        else:
            x = indenter._leftIndent - self._bulletDedent
        canv.setFont(self._bulletFontName, self._bulletFontSize)
        canv.setFillColor(self._bulletColor)
        bulletAlign = self._bulletAlign
        value = _bulletFormat(value, self._bulletType, self._bulletFormat)
        if bulletAlign == 'left':
            canv.drawString(x, y, value)
        elif bulletAlign == 'right':
            canv.drawRightString(x, y, value)
        elif bulletAlign in ('center', 'centre'):
            canv.drawCentredString(x, y, value)
        elif bulletAlign.startswith('numeric') or bulletAlign.startswith('decimal'):
            pc = bulletAlign[7:].strip() or '.'
            canv.drawAlignedString(x, y, value, pc)
        else:
            raise ValueError('Invalid bulletAlign: %r' % bulletAlign)
        canv.restoreState()