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
def _included_frames(frame_count, frame_format, frame_dir):
    return INCLUDED_FRAMES.format(Nframes=frame_count, frame_dir=frame_dir, frame_format=frame_format)