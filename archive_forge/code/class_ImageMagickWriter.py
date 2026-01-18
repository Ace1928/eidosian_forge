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
@writers.register('imagemagick')
class ImageMagickWriter(ImageMagickBase, MovieWriter):
    """
    Pipe-based animated gif writer.

    Frames are streamed directly to ImageMagick via a pipe and written
    in a single pass.
    """
    input_names = '-'