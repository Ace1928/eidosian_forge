import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
class _SlotValue:

    def __init__(self, device, slot):
        self._device = device
        self._slot = slot

    def __getitem__(self, code):
        if code.type is not libevdev.EV_ABS or code <= libevdev.EV_ABS.ABS_MT_SLOT:
            raise InvalidArgumentException('Event code must be one of EV_ABS.ABS_MT_*')
        if not self._device.has(code):
            return None
        return self._device._libevdev.slot_value(self._slot, code.value)

    def __setitem__(self, code, value):
        if code.type is not libevdev.EV_ABS or code <= libevdev.EV_ABS.ABS_MT_SLOT:
            raise InvalidArgumentException('Event code must be one of EV_ABS.ABS_MT_*')
        if not self._device.has(code):
            raise InvalidArgumentException('Event code does not exist')
        self._device._libevdev.slot_value(self._slot, code.value, new_value=value)