import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def CenterAndBox(canvas, s, cx=200, y=40):
    """tests string positioning, stringWidth, fontAscent, and fontDescent"""
    canvas.drawLine(cx, y - 30, cx, y + 30, color=yellow)
    w = canvas
    w = canvas.stringWidth(s)
    canvas.drawLine(cx - w / 2, y, cx + w / 2, y, color=red)
    canvas.drawString(s, cx - w / 2, y)
    canvas.defaultLineColor = Color(0.7, 0.7, 1.0)
    canvas.drawLine(cx - w / 2, y - 20, cx - w / 2, y + 20)
    canvas.drawLine(cx + w / 2, y - 20, cx + w / 2, y + 20)
    asc, desc = (canvas.fontAscent(), canvas.fontDescent())
    canvas.drawLine(cx - w / 2 - 20, y - asc, cx + w / 2 + 20, y - asc)
    canvas.drawLine(cx - w / 2 - 20, y + desc, cx + w / 2 + 20, y + desc)