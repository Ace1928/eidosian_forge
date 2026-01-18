import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
def _generate_transform(transform_list):
    parts = []
    for type, value in transform_list:
        if type == 'scale' and (value == (1,) or value == (1, 1)) or (type == 'translate' and value == (0, 0)) or (type == 'rotate' and value == (0,)):
            continue
        if type == 'matrix' and isinstance(value, Affine2DBase):
            value = value.to_values()
        parts.append('{}({})'.format(type, ' '.join((_short_float_fmt(x) for x in value))))
    return ' '.join(parts)