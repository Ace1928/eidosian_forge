from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
def _getFragLines(frags):
    lines = []
    cline = []
    W = frags[:]
    while W != []:
        w = W[0]
        t = w.text
        del W[0]
        i = t.find('\n')
        if i >= 0:
            tleft = t[i + 1:]
            cline.append(w.clone(text=t[:i]))
            lines.append(cline)
            cline = []
            if tleft != '':
                W.insert(0, w.clone(text=tleft))
        else:
            cline.append(w)
    if cline != []:
        lines.append(cline)
    return lines