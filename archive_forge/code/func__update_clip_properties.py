import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def _update_clip_properties(self):
    if self._bbox_patch:
        clipprops = dict(clip_box=self.clipbox, clip_path=self._clippath, clip_on=self._clipon)
        self._bbox_patch.update(clipprops)