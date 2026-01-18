import subprocess as sp
import numpy as np
from .abstract import VideoReaderAbstract, VideoWriterAbstract
from .ffprobe import ffprobe
from .. import _FFMPEG_APPLICATION
from .. import _FFMPEG_PATH
from .. import _FFMPEG_SUPPORTED_DECODERS
from .. import _FFMPEG_SUPPORTED_ENCODERS
from .. import _HAS_FFMPEG
from ..utils import *
def _probCountFrames(self):
    probecmd = [_FFMPEG_PATH + '/ffprobe'] + ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', self._filename]
    return np.int(check_output(probecmd).decode().split('\n')[0])