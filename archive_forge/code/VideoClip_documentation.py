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
Returns the of all valid entries which contain ``string`` for the
           argument ``arg`` of ``TextClip``, for instance

           >>> # Find all the available fonts which contain "Courier"
           >>> print ( TextClip.search('Courier', 'font') )

        