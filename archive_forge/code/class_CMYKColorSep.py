import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
class CMYKColorSep(CMYKColor):
    """special case color for making separating pdfs"""
    _scale = 1.0

    def __init__(self, cyan=0, magenta=0, yellow=0, black=0, spotName=None, density=1, alpha=1):
        CMYKColor.__init__(self, cyan, magenta, yellow, black, spotName, density, knockout=None, alpha=alpha)
    _cKwds = 'cyan magenta yellow black density alpha spotName'.split()