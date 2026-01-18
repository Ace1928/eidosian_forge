import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [('wFormatTag', WORD), ('nChannels', WORD), ('nSamplesPerSec', DWORD), ('nAvgBytesPerSec', DWORD), ('nBlockAlign', WORD), ('wBitsPerSample', WORD), ('cbSize', WORD)]

    def __repr__(self):
        return 'WAVEFORMATEX(wFormatTag={}, nChannels={}, nSamplesPerSec={}, nAvgBytesPersec={}, nBlockAlign={}, wBitsPerSample={}, cbSize={})'.format(self.wFormatTag, self.nChannels, self.nSamplesPerSec, self.nAvgBytesPerSec, self.nBlockAlign, self.wBitsPerSample, self.cbSize)