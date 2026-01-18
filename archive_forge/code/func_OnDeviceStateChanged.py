from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
def OnDeviceStateChanged(self, pwstrDeviceId, dwNewState):
    device = self.audio_devices.get_cached_device(pwstrDeviceId)
    old_state = device.state
    pyglet_old_state = Win32AudioDevice.platform_state[old_state]
    pyglet_new_state = Win32AudioDevice.platform_state[dwNewState]
    assert _debug(f"Audio device '{device.name}' changed state. From: {pyglet_old_state} to: {pyglet_new_state}")
    device.state = dwNewState
    self.audio_devices.dispatch_event('on_device_state_changed', device, pyglet_old_state, pyglet_new_state)