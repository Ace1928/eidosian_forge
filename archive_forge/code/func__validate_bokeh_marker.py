import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
def _validate_bokeh_marker(value):
    """Validate the markers."""
    try:
        from bokeh.core.enums import MarkerType
    except ImportError:
        MarkerType = ['asterisk', 'circle', 'circle_cross', 'circle_dot', 'circle_x', 'circle_y', 'cross', 'dash', 'diamond', 'diamond_cross', 'diamond_dot', 'dot', 'hex', 'hex_dot', 'inverted_triangle', 'plus', 'square', 'square_cross', 'square_dot', 'square_pin', 'square_x', 'star', 'star_dot', 'triangle', 'triangle_dot', 'triangle_pin', 'x', 'y']
    all_markers = list(MarkerType)
    if value not in all_markers:
        raise ValueError(f'{value} is not one of {all_markers}')
    return value