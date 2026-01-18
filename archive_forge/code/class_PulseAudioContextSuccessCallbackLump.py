import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class PulseAudioContextSuccessCallbackLump:

    def __init__(self, context: PulseAudioContext, callback: Optional[PulseAudioContextSuccessCallback]=None) -> None:
        self.pa_callback = pa.pa_context_success_cb_t(self._success_callback)
        self.py_callback = callback
        self.context = context

    def _success_callback(self, context, success, userdata):
        if self.py_callback is not None:
            self.py_callback(context, success, userdata)
        self.context.mainloop.signal()