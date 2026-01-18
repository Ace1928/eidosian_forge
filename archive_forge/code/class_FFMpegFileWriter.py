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
@writers.register('ffmpeg_file')
class FFMpegFileWriter(FFMpegBase, FileMovieWriter):
    """
    File-based ffmpeg writer.

    Frames are written to temporary files on disk and then stitched together at the end.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """
    supported_formats = ['png', 'jpeg', 'tiff', 'raw', 'rgba']

    def _args(self):
        args = []
        if self.frame_format in {'raw', 'rgba'}:
            args += ['-f', 'image2', '-vcodec', 'rawvideo', '-video_size', '%dx%d' % self.frame_size, '-pixel_format', 'rgba']
        args += ['-framerate', str(self.fps), '-i', self._base_temp_name()]
        if not self._tmpdir:
            args += ['-frames:v', str(self._frame_counter)]
        if _log.getEffectiveLevel() > logging.DEBUG:
            args += ['-loglevel', 'error']
        return [self.bin_path(), *args, *self.output_args]