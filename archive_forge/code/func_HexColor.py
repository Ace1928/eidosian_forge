import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def HexColor(val, htmlOnly=False, hasAlpha=False):
    """This function converts a hex string, or an actual integer number,
    into the corresponding color.  E.g., in "#AABBCC" or 0xAABBCC,
    AA is the red, BB is the green, and CC is the blue (00-FF).

    An alpha value can also be given in the form #AABBCCDD or 0xAABBCCDD where
    DD is the alpha value if hasAlpha is True.

    For completeness I assume that #aabbcc or 0xaabbcc are hex numbers
    otherwise a pure integer is converted as decimal rgb.  If htmlOnly is true,
    only the #aabbcc form is allowed.

    >>> HexColor('#ffffff')
    Color(1,1,1,1)
    >>> HexColor('#FFFFFF')
    Color(1,1,1,1)
    >>> HexColor('0xffffff')
    Color(1,1,1,1)
    >>> HexColor('16777215')
    Color(1,1,1,1)

    An '0x' or '#' prefix is required for hex (as opposed to decimal):

    >>> HexColor('ffffff')
    Traceback (most recent call last):
    ValueError: invalid literal for int() with base 10: 'ffffff'

    >>> HexColor('#FFFFFF', htmlOnly=True)
    Color(1,1,1,1)
    >>> HexColor('0xffffff', htmlOnly=True)
    Traceback (most recent call last):
    ValueError: not a hex string
    >>> HexColor('16777215', htmlOnly=True)
    Traceback (most recent call last):
    ValueError: not a hex string

    """
    if isStr(val):
        val = asNative(val)
        b = 10
        if val[:1] == '#':
            val = val[1:]
            b = 16
            if len(val) == 8:
                alpha = True
        else:
            if htmlOnly:
                raise ValueError('not a hex string')
            if val[:2].lower() == '0x':
                b = 16
                val = val[2:]
                if len(val) == 8:
                    alpha = True
        val = int(val, b)
    if hasAlpha:
        return Color((val >> 24 & 255) / 255.0, (val >> 16 & 255) / 255.0, (val >> 8 & 255) / 255.0, (val & 255) / 255.0)
    return Color((val >> 16 & 255) / 255.0, (val >> 8 & 255) / 255.0, (val & 255) / 255.0)