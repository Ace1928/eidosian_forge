from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
class XAudio2VoiceCallback(com.COMObject):
    """Callback class used to trigger when buffers or streams end.
           WARNING: Whenever a callback is running, XAudio2 cannot generate audio.
           Make sure these functions run as fast as possible and do not block/delay more than a few milliseconds.
           MS Recommendation:
           At a minimum, callback functions must not do the following:
                - Access the hard disk or other permanent storage
                - Make expensive or blocking API calls
                - Synchronize with other parts of client code
                - Require significant CPU usage
    """
    _interfaces_ = [lib.IXAudio2VoiceCallback]

    def __init__(self):
        super().__init__()
        self.on_buffer_end = None

    def OnBufferEnd(self, pBufferContext):
        self.on_buffer_end(pBufferContext)

    def OnVoiceError(self, pBufferContext, hresult):
        raise Exception(f'Error occurred during audio playback: {hresult}')