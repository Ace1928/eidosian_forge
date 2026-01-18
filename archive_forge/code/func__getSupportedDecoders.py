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
def _getSupportedDecoders(self):
    return _FFMPEG_SUPPORTED_DECODERS