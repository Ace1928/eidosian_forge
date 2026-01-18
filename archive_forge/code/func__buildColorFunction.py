import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def _buildColorFunction(colors, positions):
    from reportlab.pdfbase.pdfdoc import PDFExponentialFunction, PDFStitchingFunction
    if positions is not None and len(positions) != len(colors):
        raise ValueError('need to have the same number of colors and positions')
    if len(colors) == 1:
        return PDFExponentialFunction(N=1, C0=colors[0], C1=colors[0])
    if len(colors) == 2:
        if positions is None or (positions[0] == 0 and positions[1] == 1):
            return PDFExponentialFunction(N=1, C0=colors[0], C1=colors[1])
    if positions is None:
        nc = len(colors)
        positions = [float(x) / (nc - 1) for x in range(nc)]
    else:
        poscolors = list(zip(positions, colors))
        poscolors.sort(key=lambda x: x[0])
        if poscolors[0][0] != 0:
            poscolors.insert(0, (0.0, poscolors[0][1]))
        if poscolors[-1][0] != 1:
            poscolors.append((1.0, poscolors[-1][1]))
        positions, colors = list(zip(*poscolors))
    functions = []
    bounds = [pos for pos in positions[1:-1]]
    encode = []
    lastcolor = colors[0]
    for color in colors[1:]:
        functions.append(PDFExponentialFunction(N=1, C0=lastcolor, C1=color))
        lastcolor = color
        encode.append(0.0)
        encode.append(1.0)
    return PDFStitchingFunction(functions, bounds, encode, Domain='[0.0 1.0]')