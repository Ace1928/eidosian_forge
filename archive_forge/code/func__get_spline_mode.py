import functools
import math
import operator
import textwrap
import cupy
def _get_spline_mode(mode):
    """spline boundary mode for interpolation with order >= 2."""
    if mode in ['mirror', 'reflect', 'grid-wrap']:
        return mode
    elif mode == 'grid-mirror':
        return 'reflect'
    return 'reflect' if mode == 'nearest' else 'mirror'