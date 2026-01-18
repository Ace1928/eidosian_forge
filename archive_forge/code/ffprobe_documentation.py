import subprocess as sp
from ..utils import *
from .. import _HAS_FFMPEG
from .. import _FFMPEG_PATH
from .. import _FFPROBE_APPLICATION
get metadata by using ffprobe

    Checks the output of ffprobe on the desired video
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

    