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
def _update_glyph_map_defs(self, glyph_map_new):
    """
        Emit definitions for not-yet-defined glyphs, and record them as having
        been defined.
        """
    writer = self.writer
    if glyph_map_new:
        writer.start('defs')
        for char_id, (vertices, codes) in glyph_map_new.items():
            char_id = self._adjust_char_id(char_id)
            path_data = self._convert_path(Path(vertices * 64, codes), simplify=False)
            writer.element('path', id=char_id, d=path_data, transform=_generate_transform([('scale', (1 / 64,))]))
        writer.end('defs')
        self._glyph_map.update(glyph_map_new)