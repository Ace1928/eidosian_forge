import os
import subprocess as sp
import tempfile
import warnings
import numpy as np
import proglog
from imageio import imread, imsave
from ..Clip import Clip
from ..compat import DEVNULL, string_types
from ..config import get_setting
from ..decorators import (add_mask_if_none, apply_to_mask,
from ..tools import (deprecated_version_of, extensions_dict, find_extension,
from .io.ffmpeg_writer import ffmpeg_write_video
from .io.gif_writers import (write_gif, write_gif_with_image_io,
from .tools.drawing import blit
class ColorClip(ImageClip):
    """An ImageClip showing just one color.

    Parameters
    -----------

    size
      Size (width, height) in pixels of the clip.

    color
      If argument ``ismask`` is False, ``color`` indicates
      the color in RGB of the clip (default is black). If `ismask``
      is True, ``color`` must be  a float between 0 and 1 (default is 1)

    ismask
      Set to true if the clip will be used as a mask.

    col
      Has been deprecated. Do not use.
    """

    def __init__(self, size, color=None, ismask=False, duration=None, col=None):
        if col is not None:
            warnings.warn('The `ColorClip` parameter `col` has been deprecated. Please use `color` instead.', DeprecationWarning)
            if color is not None:
                warnings.warn('The arguments `color` and `col` have both been passed to `ColorClip` so `col` has been ignored.', UserWarning)
            else:
                color = col
        w, h = size
        shape = (h, w) if np.isscalar(color) else (h, w, len(color))
        ImageClip.__init__(self, np.tile(color, w * h).reshape(shape), ismask=ismask, duration=duration)