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
def _update_converter():
    try:
        mpl._get_executable_info('gs')
    except mpl.ExecutableNotFoundError:
        pass
    else:
        converter['pdf'] = converter['eps'] = _GSConverter()
    try:
        mpl._get_executable_info('inkscape')
    except mpl.ExecutableNotFoundError:
        pass
    else:
        converter['svg'] = _SVGConverter()