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
def _make_id(self, type, content):
    salt = mpl.rcParams['svg.hashsalt']
    if salt is None:
        salt = str(uuid.uuid4())
    m = hashlib.sha256()
    m.update(salt.encode('utf8'))
    m.update(str(content).encode('utf8'))
    return f'{type}{m.hexdigest()[:10]}'