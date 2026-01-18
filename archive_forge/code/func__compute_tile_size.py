from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def _compute_tile_size(shape):
    slices = shape[0]
    approx_rows = int(round(math.sqrt(slices)))
    image_width = max(min_texture_width, min(max_texture_width, utils.next_power_of_2(approx_rows * shape[1])))
    columns = image_width // shape[2]
    rows = int(math.ceil(slices / columns))
    image_height = max(min_texture_width, utils.next_power_of_2(rows * shape[1]))
    return (rows, columns, image_width, image_height)