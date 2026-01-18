from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
def _query_all_devices(self) -> List[Win32AudioDevice]:
    return self.get_devices(flow=eRender, state=DEVICE_STATEMASK_ALL) + self.get_devices(flow=eCapture, state=DEVICE_STATEMASK_ALL)