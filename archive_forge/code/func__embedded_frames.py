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
def _embedded_frames(frame_list, frame_format):
    """frame_list should be a list of base64-encoded png files"""
    if frame_format == 'svg':
        frame_format = 'svg+xml'
    template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'
    return '\n' + ''.join((template.format(i, frame_format, frame_data.replace('\n', '\\\n')) for i, frame_data in enumerate(frame_list)))