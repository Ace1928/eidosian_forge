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
def _get_clip_attrs(self, gc):
    cliprect = gc.get_clip_rectangle()
    clippath, clippath_trans = gc.get_clip_path()
    if clippath is not None:
        clippath_trans = self._make_flip_transform(clippath_trans)
        dictkey = (id(clippath), str(clippath_trans))
    elif cliprect is not None:
        x, y, w, h = cliprect.bounds
        y = self.height - (y + h)
        dictkey = (x, y, w, h)
    else:
        return {}
    clip = self._clipd.get(dictkey)
    if clip is None:
        oid = self._make_id('p', dictkey)
        if clippath is not None:
            self._clipd[dictkey] = ((clippath, clippath_trans), oid)
        else:
            self._clipd[dictkey] = (dictkey, oid)
    else:
        clip, oid = clip
    return {'clip-path': f'url(#{oid})'}