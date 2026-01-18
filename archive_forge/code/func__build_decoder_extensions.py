import os
import platform
import warnings
from pyglet import image
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
@staticmethod
def _build_decoder_extensions():
    """Extension support varies depending on OS version."""
    extensions = []
    if WINDOWS_VISTA_OR_GREATER:
        extensions.extend(['.asf', '.wma', '.wmv', '.mp3', '.sami', '.smi'])
    if WINDOWS_7_OR_GREATER:
        extensions.extend(['.3g2', '.3gp', '.3gp2', '.3gp', '.aac', '.adts', '.avi', '.m4a', '.m4v'])
    if WINDOWS_10_ANNIVERSARY_UPDATE_OR_GREATER:
        extensions.extend(['.flac'])
    return extensions