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
def _setup_blit(self):
    self._blit_cache = dict()
    self._drawn_artists = []
    self._post_draw(None, self._blit)
    self._init_draw()
    self._resize_id = self._fig.canvas.mpl_connect('resize_event', self._on_resize)