import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
def _create_wave_format(audio_format):
    wfx = lib.WAVEFORMATEX()
    wfx.wFormatTag = lib.WAVE_FORMAT_PCM
    wfx.nChannels = audio_format.channels
    wfx.nSamplesPerSec = audio_format.sample_rate
    wfx.wBitsPerSample = audio_format.sample_size
    wfx.nBlockAlign = wfx.wBitsPerSample * wfx.nChannels // 8
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign
    return wfx