import itertools
import io
import base64
import numpy as np
import warnings
import matplotlib
from matplotlib.colors import colorConverter
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import ticker
def _many_to_one(input_dict):
    """Convert a many-to-one mapping to a one-to-one mapping"""
    return dict(((key, val) for keys, val in input_dict.items() for key in keys))