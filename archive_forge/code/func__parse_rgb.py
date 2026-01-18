import collections
import re
from colorsys import hls_to_rgb
from .parser import parse_one_component_value
def _parse_rgb(args, alpha):
    """Parse a list of RGB channels.

    If args is a list of 3 INTEGER tokens or 3 PERCENTAGE tokens, return RGB
    values as a tuple of 3 floats in 0..1. Otherwise, return None.

    """
    types = [arg.type for arg in args]
    if types == ['number', 'number', 'number'] and all((a.is_integer for a in args)):
        r, g, b = [arg.int_value / 255 for arg in args[:3]]
        return RGBA(r, g, b, alpha)
    elif types == ['percentage', 'percentage', 'percentage']:
        r, g, b = [arg.value / 100 for arg in args[:3]]
        return RGBA(r, g, b, alpha)