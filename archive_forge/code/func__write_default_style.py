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
def _write_default_style(self):
    writer = self.writer
    default_style = _generate_css({'stroke-linejoin': 'round', 'stroke-linecap': 'butt'})
    writer.start('defs')
    writer.element('style', type='text/css', text='*{%s}' % default_style)
    writer.end('defs')