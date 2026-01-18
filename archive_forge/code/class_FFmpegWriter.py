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
class FFmpegWriter(VideoWriterAbstract):
    """Writes frames using FFmpeg

    Using FFmpeg as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, 'Cannot find installation of real FFmpeg (which comes with ffprobe).'
        super(FFmpegWriter, self).__init__(*args, **kwargs)

    def _getSupportedEncoders(self):
        return _FFMPEG_SUPPORTED_ENCODERS

    def _createProcess(self, inputdict, outputdict, verbosity):
        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)
        cmd = [_FFMPEG_PATH + '/' + _FFMPEG_APPLICATION, '-y'] + iargs + ['-i', '-'] + oargs + [self._filename]
        self._cmd = ' '.join(cmd)
        if self.verbosity > 0:
            print(self._cmd)
            self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
        else:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=self.DEVNULL, stderr=sp.STDOUT)