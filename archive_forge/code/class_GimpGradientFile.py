from __future__ import annotations
from math import log, pi, sin, sqrt
from ._binary import o8
class GimpGradientFile(GradientFile):
    """File handler for GIMP's gradient format."""

    def __init__(self, fp):
        if fp.readline()[:13] != b'GIMP Gradient':
            msg = 'not a GIMP gradient file'
            raise SyntaxError(msg)
        line = fp.readline()
        if line.startswith(b'Name: '):
            line = fp.readline().strip()
        count = int(line)
        gradient = []
        for i in range(count):
            s = fp.readline().split()
            w = [float(x) for x in s[:11]]
            x0, x1 = (w[0], w[2])
            xm = w[1]
            rgb0 = w[3:7]
            rgb1 = w[7:11]
            segment = SEGMENTS[int(s[11])]
            cspace = int(s[12])
            if cspace != 0:
                msg = 'cannot handle HSV colour space'
                raise OSError(msg)
            gradient.append((x0, x1, xm, rgb0, rgb1, segment))
        self.gradient = gradient