import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
class toColor:
    """Accepot an expression returnng a Color subclass.

    This used to accept arbitrary Python expressions, which resulted in increasngly devilish CVEs and
    security holes from tie to time.  In April 2023 we are creating explicit, "dumb" parsing code to
    replace this.  Acceptable patterns are

    a Color instance passed in by the Python programmer
    a named list of colours ('pink' etc')
    list of 3 or 4 numbers
    all CSS colour expression
    """
    _G = {}

    def __init__(self):
        self.extraColorsNS = {}

    def setExtraColorsNameSpace(self, NS):
        self.extraColorsNS = NS

    def __call__(self, arg, default=None):
        """try to map an arbitrary arg to a color instance
        """
        if isinstance(arg, Color):
            return arg
        if isinstance(arg, (tuple, list)):
            assert 3 <= len(arg) <= 4, 'Can only convert 3 and 4 sequences to color'
            assert 0 <= min(arg) and max(arg) <= 1
            return len(arg) == 3 and Color(arg[0], arg[1], arg[2]) or CMYKColor(arg[0], arg[1], arg[2], arg[3])
        elif isStr(arg):
            arg = asNative(arg)
            C = cssParse(arg)
            if C:
                return C
            if arg in self.extraColorsNS:
                return self.extraColorsNS[arg]
            C = getAllNamedColors()
            s = arg.lower()
            if s in C:
                return C[s]
            pat = re.compile('(Blacker|Whiter)\\((\\w+)\\,\\s?([0-9.]+)\\)')
            m = pat.match(arg)
            if m:
                funcname, rootcolor, num = m.groups()
                if funcname == 'Blacker':
                    return Blacker(rootcolor, float(num))
                else:
                    return Whiter(rootcolor, float(num))
            try:
                import ast
                expr = ast.literal_eval(arg)
                return toColor(expr)
            except (SyntaxError, ValueError):
                pass
            if rl_config.toColorCanUse == 'rl_safe_eval':
                G = C.copy()
                G.update(self.extraColorsNS)
                if not self._G:
                    C = globals()
                    self._G = {s: C[s] for s in 'Blacker CMYKColor CMYKColorSep Color ColorType HexColor PCMYKColor PCMYKColorSep Whiter\n                        _chooseEnforceColorSpace _enforceCMYK _enforceError _enforceRGB _enforceSEP _enforceSEP_BLACK\n                        _enforceSEP_CMYK _namedColors _re_css asNative cmyk2rgb cmykDistance color2bw colorDistance\n                        cssParse describe fade fp_str getAllNamedColors hsl2rgb hue2rgb isStr linearlyInterpolatedColor\n                        literal_eval obj_R_G_B opaqueColor rgb2cmyk setColors toColor toColorOrNone'.split()}
                G.update(self._G)
                try:
                    return toColor(rl_safe_eval(arg, g=G, l={}))
                except:
                    pass
            elif rl_config.toColorCanUse == 'rl_extended_literal_eval':
                C = globals()
                S = getAllNamedColors().copy()
                C = {k: C[k] for k in 'Blacker CMYKColor CMYKColorSep Color ColorType HexColor PCMYKColor PCMYKColorSep Whiter\n                        _chooseEnforceColorSpace _enforceCMYK _enforceError _enforceRGB _enforceSEP _enforceSEP_BLACK\n                        _enforceSEP_CMYK _namedColors _re_css asNative cmyk2rgb cmykDistance color2bw colorDistance\n                        cssParse describe fade fp_str getAllNamedColors hsl2rgb hue2rgb linearlyInterpolatedColor\n                        obj_R_G_B opaqueColor rgb2cmyk setColors toColor toColorOrNone'.split() if callable(C.get(k, None))}
                try:
                    return rl_extended_literal_eval(arg, C, S)
                except (ValueError, SyntaxError):
                    pass
        try:
            return HexColor(arg)
        except:
            if default is None:
                raise ValueError('Invalid color value %r' % arg)
            return default