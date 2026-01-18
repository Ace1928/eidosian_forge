from __future__ import annotations
import importlib
import math
from functools import wraps
from string import ascii_letters
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.diverging
from matplotlib import cm, colors
from pymatgen.core import Element
def _decide_fontcolor(rgba: tuple) -> Literal['black', 'white']:
    red, green, blue, _ = rgba
    if red * 0.299 + green * 0.587 + blue * 0.114 > 186 / 255:
        return 'black'
    return 'white'