import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def _set_extremes(self):
    if self._rgba_under:
        self._lut[self._i_under] = self._rgba_under
    else:
        self._lut[self._i_under] = self._lut[0]
    if self._rgba_over:
        self._lut[self._i_over] = self._rgba_over
    else:
        self._lut[self._i_over] = self._lut[self.N - 1]
    self._lut[self._i_bad] = self._rgba_bad