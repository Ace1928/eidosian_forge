import atexit
import functools
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import weakref
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook
from matplotlib.testing.exceptions import ImageComparisonFailure
class _SVGWithMatplotlibFontsConverter(_SVGConverter):
    """
    A SVG converter which explicitly adds the fonts shipped by Matplotlib to
    Inkspace's font search path, to better support `svg.fonttype = "none"`
    (which is in particular used by certain mathtext tests).
    """

    def __call__(self, orig, dest):
        if not hasattr(self, '_tmpdir'):
            self._tmpdir = TemporaryDirectory()
            shutil.copytree(cbook._get_data_path('fonts/ttf'), Path(self._tmpdir.name, 'fonts'))
        return super().__call__(orig, dest)