import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
def _get_clip_cmd(self, gc):
    clip = []
    rect = gc.get_clip_rectangle()
    if rect is not None:
        clip.append(f'{_nums_to_str(*rect.p0, *rect.size)} rectclip\n')
    path, trf = gc.get_clip_path()
    if path is not None:
        key = (path, id(trf))
        custom_clip_cmd = self._clip_paths.get(key)
        if custom_clip_cmd is None:
            custom_clip_cmd = 'c%d' % len(self._clip_paths)
            self._pswriter.write(f'/{custom_clip_cmd} {{\n{self._convert_path(path, trf, simplify=False)}\nclip\nnewpath\n}} bind def\n')
            self._clip_paths[key] = custom_clip_cmd
        clip.append(f'{custom_clip_cmd}\n')
    return ''.join(clip)