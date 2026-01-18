import collections
import re
from colorsys import hls_to_rgb
from .parser import parse_one_component_value
def _parse_alpha(args):
    """Parse a list of one alpha value.

    If args is a list of a single INTEGER or NUMBER token,
    return its value clipped to the 0..1 range. Otherwise, return None.

    """
    if len(args) == 1 and args[0].type == 'number':
        return min(1, max(0, args[0].value))