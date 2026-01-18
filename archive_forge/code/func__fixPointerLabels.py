import functools
from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Ellipse, Wedge, String, STATE_DEFAULTS, ArcPath, Polygon, Rect, PolyLine, Line
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.textlabels import Label
from reportlab import cmp
from reportlab.graphics.charts.utils3d import _getShaded, _2rad, _360, _180_pi
def _fixPointerLabels(n, L, x, y, width, height, side=None):
    LR = ([], [])
    mlr = [0, 0]
    for l in L:
        i, w = _fPLSide(l, width, side)
        LR[i].append(l)
        mlr[i] = max(w, mlr[i])
    mul = 1
    G = n * [None]
    mel = 0
    hh = height * 0.5
    yhh = y + hh
    m = max(mlr)
    for i in (0, 1):
        T = LR[i]
        if T:
            B = []
            aB = B.append
            S = []
            aS = S.append
            T.sort(key=_fPLCF)
            p = 0
            yh = y + height
            for l in T:
                data = l._origdata
                inc = x + mul * (m - data['width'])
                l.x += inc
                G[data['index']] = l
                ly = yhh + data['smid'] * hh
                b = data['bounds']
                b2 = (b[3] - b[1]) * 0.5
                if ly + b2 > yh:
                    ly = yh - b2
                if ly - b2 < y:
                    ly = y + b2
                data['bounds'] = b = (b[0], ly - b2, b[2], ly + b2)
                aB(b)
                l.y = ly
                aS(max(0, yh - ly - b2))
                yh = ly - b2
                p = max(p, data['edgePad'] + data['piePad'])
                mel = max(mel, abs(data['smid'] * (hh + data['elbowLength'])) - hh)
            aS(yh - y)
            iter = 0
            nT = len(T)
            while iter < 30:
                R = findOverlapRun(B, wrap=0)
                if not R:
                    break
                nR = len(R)
                if nR == nT:
                    break
                j0 = R[0]
                j1 = R[-1]
                jl = j1 + 1
                sAbove = sum(S[:j0 + 1])
                sFree = sAbove + sum(S[jl:])
                sNeed = sum([b[3] - b[1] for b in B[j0:jl]]) + jl - j0 - (B[j0][3] - B[j1][1])
                if sNeed > sFree:
                    break
                yh = B[j0][3] + sAbove * sNeed / sFree
                for r in R:
                    l = T[r]
                    data = l._origdata
                    b = data['bounds']
                    b2 = (b[3] - b[1]) * 0.5
                    yh -= 0.5
                    ly = l.y = yh - b2
                    B[r] = data['bounds'] = (b[0], ly - b2, b[2], yh)
                    yh = ly - b2 - 0.5
            mlr[i] = m + p
        mul = -1
    return (G, mlr[0], mlr[1], mel)