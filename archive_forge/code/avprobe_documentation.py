import subprocess as sp
from ..utils import *
from .. import _HAS_AVCONV, _LIBAV_MAJOR_VERSION
from .. import _AVCONV_PATH
from .. import _AVPROBE_APPLICATION
import json
get metadata by using avprobe

    Checks the output of avprobe on the desired video
    file. MetaData is then parsed into a dictionary.

    Parameters
    ----------
    filename : string
        Path to the video file

    Returns
    -------
    metaDict : dict
       Dictionary containing all header-based information 
       about the passed-in source video.

    