import subprocess as sp
import numpy as np
from .abstract import VideoReaderAbstract, VideoWriterAbstract
from .avprobe import avprobe
from .. import _AVCONV_APPLICATION
from .. import _AVCONV_PATH
from .. import _HAS_AVCONV
from ..utils import *
class LibAVWriter(VideoWriterAbstract):
    """Writes frames using libav

    Using libav as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_AVCONV, 'Cannot find installation of libav (which comes with avprobe).'
        super(LibAVWriter, self).__init__(*args, **kwargs)

    def _createProcess(self, inputdict, outputdict, verbosity):
        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)
        cmd = [_AVCONV_PATH + '/avconv', '-y'] + iargs + ['-i', 'pipe:'] + oargs + [self._filename]
        self._cmd = ' '.join(cmd)
        if self.verbosity == 0:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        else:
            print(cmd)
            self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)

    def _gray2RGB(self, data):
        T, M, N, C = data.shape
        if C < 3:
            vid = np.empty((T, M, N, C + 2), dtype=data.dtype)
            vid[:, :, :, 0] = data[:, :, :, 0]
            vid[:, :, :, 1] = data[:, :, :, 0]
            vid[:, :, :, 2] = data[:, :, :, 0]
            if C == 2:
                vid[:, :, :, 3] = data[:, :, :, 1]
            return vid
        return data

    def _warmStart(self, M, N, C, dtype):
        if (C == 2 or C == 4) and dtype.itemsize == 2 and ('-pix_fmt' not in self.inputdict or self.inputdict['-pix_fmt'][0:6] == 'rgba64'):
            raise ValueError('libAV doesnt support rgba64 formats')
        if C < 3 and '-pix_fmt' not in self.inputdict:
            C += 2
            self._prepareData = self._gray2RGB
        super(LibAVWriter, self)._warmStart(M, N, C, dtype)