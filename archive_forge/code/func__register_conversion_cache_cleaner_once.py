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
@functools.cache
def _register_conversion_cache_cleaner_once():
    atexit.register(_clean_conversion_cache)