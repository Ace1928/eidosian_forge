import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
class cssParse:

    def pcVal(self, v):
        v = v.strip()
        try:
            c = float(v[:-1])
            c = min(100, max(0, c)) / 100.0
        except:
            raise ValueError('bad percentage argument value %r in css color %r' % (v, self.s))
        return c

    def rgbPcVal(self, v):
        return int(self.pcVal(v) * 255 + 0.5) / 255.0

    def rgbVal(self, v):
        v = v.strip()
        try:
            c = float(v)
            if 0 < c <= 1:
                c *= 255
            return int(min(255, max(0, c))) / 255.0
        except:
            raise ValueError('bad argument value %r in css color %r' % (v, self.s))

    def hueVal(self, v):
        v = v.strip()
        try:
            c = float(v)
            return (c % 360 + 360) % 360 / 360.0
        except:
            raise ValueError('bad hue argument value %r in css color %r' % (v, self.s))

    def alphaVal(self, v, c=1, n='alpha'):
        try:
            a = float(v)
            return min(c, max(0, a))
        except:
            raise ValueError('bad %s argument value %r in css color %r' % (n, v, self.s))
    _n_c = dict(pcmyk=(4, 100, True, False), cmyk=(4, 1, True, False), hsl=(3, 1, False, True), rgb=(3, 1, False, False))

    def __call__(self, s):
        n = _re_css.match(s)
        if not n:
            return
        self.s = s
        b, c, cmyk, hsl = self._n_c[n.group(1)]
        ha = n.group(2)
        n = n.group(3).split(',')
        if len(n) != b + (ha and 1 or 0):
            raise ValueError('css color %r has wrong number of components' % s)
        if ha:
            n, a = (n[:b], self.alphaVal(n[b], c))
        else:
            a = c
        if cmyk:
            C = self.alphaVal(n[0], c, 'cyan')
            M = self.alphaVal(n[1], c, 'magenta')
            Y = self.alphaVal(n[2], c, 'yellow')
            K = self.alphaVal(n[3], c, 'black')
            return (c > 1 and PCMYKColor or CMYKColor)(C, M, Y, K, alpha=a)
        else:
            if hsl:
                R, G, B = hsl2rgb(self.hueVal(n[0]), self.pcVal(n[1]), self.pcVal(n[2]))
            else:
                R, G, B = list(map('%' in n[0] and self.rgbPcVal or self.rgbVal, n))
            return Color(R, G, B, a)