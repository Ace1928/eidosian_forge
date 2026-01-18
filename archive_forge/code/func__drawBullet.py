from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
def _drawBullet(canvas, offset, cur_y, bulletText, style, rtl):
    """draw a bullet text could be a simple string or a frag list"""
    bulletAnchor = style.bulletAnchor
    if rtl or style.bulletAnchor != 'start':
        numeric = bulletAnchor == 'numeric'
        if isStr(bulletText):
            t = bulletText
            q = numeric and decimalSymbol in t
            if q:
                t = t[:t.index(decimalSymbol)]
            bulletWidth = stringWidth(t, style.bulletFontName, style.bulletFontSize)
            if q:
                bulletWidth += 0.5 * stringWidth(decimalSymbol, style.bulletFontName, style.bulletFontSize)
        else:
            bulletWidth = 0
            for f in bulletText:
                t = f.text
                q = numeric and decimalSymbol in t
                if q:
                    t = t[:t.index(decimalSymbol)]
                    bulletWidth += 0.5 * stringWidth(decimalSymbol, f.fontName, f.fontSize)
                bulletWidth += stringWidth(t, f.fontName, f.fontSize)
                if q:
                    break
    else:
        bulletWidth = 0
    if bulletAnchor == 'middle':
        bulletWidth *= 0.5
    cur_y += getattr(style, 'bulletOffsetY', 0)
    if not rtl:
        tx2 = canvas.beginText(style.bulletIndent - bulletWidth, cur_y)
    else:
        width = rtl[0]
        bulletStart = width + style.rightIndent - (style.bulletIndent + bulletWidth)
        tx2 = canvas.beginText(bulletStart, cur_y)
    tx2.setFont(style.bulletFontName, style.bulletFontSize)
    tx2.setFillColor(getattr(style, 'bulletColor', style.textColor))
    if isStr(bulletText):
        tx2.textOut(bulletText)
    else:
        for f in bulletText:
            tx2.setFont(f.fontName, f.fontSize)
            tx2.setFillColor(f.textColor)
            tx2.textOut(f.text)
    canvas.drawText(tx2)
    if not rtl:
        bulletEnd = tx2.getX() + style.bulletFontSize * 0.6
        offset = max(offset, bulletEnd - style.leftIndent)
    return offset