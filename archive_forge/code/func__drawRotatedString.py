from wxPython.wx import *
from rdkit.sping import pid as sping_pid
def _drawRotatedString(self, lines, x, y, font=None, color=None, angle=0):
    import math
    if font is None:
        font = sping_pid.Font(face='helvetica')
        self._setWXfont(font)
    ascent = self.fontAscent(font)
    height = self.fontHeight(font)
    rad = angle * math.pi / 180.0
    s = math.sin(rad)
    c = math.cos(rad)
    dx = s * height
    dy = c * height
    lx = x - dx
    ly = y - c * ascent
    for i in range(0, len(lines)):
        self.dc.DrawRotatedText(lines[i], lx + i * dx, ly + i * dy, angle)