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
def _blit_draw(self, artists):
    updated_ax = {a.axes for a in artists}
    for ax in updated_ax:
        cur_view = ax._get_view()
        view, bg = self._blit_cache.get(ax, (object(), None))
        if cur_view != view:
            self._blit_cache[ax] = (cur_view, ax.figure.canvas.copy_from_bbox(ax.bbox))
    for a in artists:
        a.axes.draw_artist(a)
    for ax in updated_ax:
        ax.figure.canvas.blit(ax.bbox)