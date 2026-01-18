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
def _getFragWords(frags, maxWidth=None):
    """ given a Parafrag list return a list of fragwords
        [[size, (f00,w00), ..., (f0n,w0n)],....,[size, (fm0,wm0), ..., (f0n,wmn)]]
        each pair f,w represents a style and some string
        each sublist represents a word
    """

    def _rescaleFrag(f):
        w = f[0]
        if isinstance(w, _PCT):
            if w._normalizer != maxWidth:
                w._normalizer = maxWidth
                w = w.normalizedValue(maxWidth)
                f[0] = w
    R = []
    aR = R.append
    W = []
    if _processed_frags(frags):
        aW = W.append
        if True:
            for f in frags:
                if isinstance(f, _InjectedFrag):
                    continue
                _rescaleFrag(f)
                if isinstance(f, _SplitFrag):
                    aW(f)
                    if isinstance(f, _HSFrag):
                        aR(_rejoinSplitFragWords(W))
                        del W[:]
                else:
                    if W:
                        aR(_rejoinSplitFragWords(W))
                        del W[:]
                    aR(f)
            if W:
                aR(_rejoinSplitFragWords(W))
        else:
            for f in frags:
                if isinstance(f, _InjectedFrag):
                    continue
                _rescaleFrag(f)
                if isinstance(f, _SplitFrag):
                    f0 = f[0]
                    if not W:
                        Wlen = 0
                        sty = None
                    elif isinstance(lf, _SplitFragHY):
                        sty, t = W[-1]
                        Wlen -= stringWidth(t[-1], sty.fontName, sty.fontSize) + 1e-08
                        W[-1] = (sty, _shyUnsplit(t))
                    Wlen += f0
                    for ts, t in f[1:]:
                        if ts is sty:
                            W[-1] = (sty, _shyUnsplit(W[-1][1], t))
                        else:
                            aW((ts, t))
                            sty = ts
                    if isinstance(f, _HSFrag):
                        lf = None
                        aR(_reconstructSplitFrags(f)([Wlen] + W))
                        del W[:]
                    else:
                        lf = f
                else:
                    if W:
                        aR(_reconstructSplitFrags(f)([Wlen] + W))
                        del W[:]
                    aR(f)
            if W:
                aR(_reconstructSplitFrags(lf)([Wlen] + W))
    else:
        hangingSpace = False
        n = 0
        hangingStrip = True
        shyIndices = False
        for f in frags:
            text = f.text
            if text != '':
                f._fkind = _FK_TEXT
                if hangingStrip:
                    text = lstrip(text)
                    if not text:
                        continue
                    hangingStrip = False
                S = split(text)
                if text[0] in whitespace or not S:
                    if W:
                        W.insert(0, n)
                        aR(_SHYWord(W) if shyIndices else W)
                        whs = hangingSpace
                        W = []
                        shyIndices = False
                        hangingSpace = False
                        n = 0
                    else:
                        whs = R and isinstance(R[-1], _HSFrag)
                    if not whs:
                        S.insert(0, '')
                    elif not S:
                        continue
                for w in S[:-1]:
                    if _shy in w:
                        w = _SHYIndexedStr(w)
                        shyIndices = True
                    W.append((f, w))
                    n += stringWidth(w, f.fontName, f.fontSize)
                    W.insert(0, n)
                    aR(_SHYWordHS(W) if shyIndices or isinstance(W, _SHYWord) else _HSFrag(W))
                    W = []
                    shyIndices = False
                    n = 0
                hangingSpace = False
                w = S[-1]
                if _shy in w:
                    w = _SHYIndexedStr(w)
                    shyIndices = True
                W.append((f, w))
                n += stringWidth(w, f.fontName, f.fontSize)
                if text and text[-1] in whitespace:
                    W.insert(0, n)
                    aR(_SHYWord(W) if shyIndices or isinstance(W, _SHYWord) else _HSFrag(W))
                    W = []
                    shyIndices = False
                    n = 0
            elif hasattr(f, 'cbDefn'):
                cb = f.cbDefn
                w = getattr(cb, 'width', 0)
                if w:
                    if hasattr(w, 'normalizedValue'):
                        w._normalizer = maxWidth
                        w = w.normalizedValue(maxWidth)
                    if W:
                        W.insert(0, n)
                        aR(_HSFrag(W) if hangingSpace else W)
                        W = []
                        shyIndices = False
                        hangingSpace = False
                        n = 0
                    f._fkind = _FK_IMG
                    aR([w, (f, '')])
                    hangingStrip = False
                else:
                    f._fkind = _FK_APPEND
                    if not W and R and isinstance(R[-1], _HSFrag):
                        R[-1].append((f, ''))
                    else:
                        W.append((f, ''))
            elif hasattr(f, 'lineBreak'):
                if W:
                    W.insert(0, n)
                    aR(W)
                    W = []
                    n = 0
                    shyIndices = False
                    hangingSpace = False
                f._fkind = _FK_BREAK
                aR([0, (f, '')])
                hangingStrip = True
        if W:
            W.insert(0, n)
            aR(_SHYWord(W) if shyIndices or isinstance(W, _SHYWord) else W)
    if not R:
        if frags:
            f = frags[0]
            f._fkind = _FK_TEXT
            R = [[0, (f, u'')]]
    return R