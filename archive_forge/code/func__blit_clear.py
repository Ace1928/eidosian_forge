import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
def _blit_clear(self, artists):
    axes = {a.axes for a in artists}
    for ax in axes:
        try:
            view, bg = self._blit_cache[ax]
        except KeyError:
            continue
        if ax._get_view() == view:
            ax.figure.canvas.restore_region(bg)
        else:
            self._blit_cache.pop(ax)