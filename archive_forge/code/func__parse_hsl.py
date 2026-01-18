import collections
import re
from colorsys import hls_to_rgb
from .parser import parse_one_component_value
def _parse_hsl(args, alpha):
    """Parse a list of HSL channels.

    If args is a list of 1 INTEGER token and 2 PERCENTAGE tokens, return RGB
    values as a tuple of 3 floats in 0..1. Otherwise, return None.

    """
    types = [arg.type for arg in args]
    if types == ['number', 'percentage', 'percentage'] and args[0].is_integer:
        r, g, b = hls_to_rgb(args[0].int_value / 360, args[2].value / 100, args[1].value / 100)
        return RGBA(r, g, b, alpha)