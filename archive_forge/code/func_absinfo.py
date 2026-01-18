import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
def absinfo(self, code, new_values=None, kernel=False):
    """
        :param code: the ABS_<*> code as integer or as string
        :param new_values: a dict with the same keys as the return values.
        :param kernel: If True, assigning new values corresponds to ``libevdev_kernel_set_abs_info``
        :return: a dictionary with the keys "value", "min", "max",
                 "resolution", "fuzz", "flat"; ``None`` if the code does not exist on
                 this device

        :note: The returned value is a copy of the value returned by
               libevdev. Changing a value in the dictionary does not change the
               matching property. To change the device, reassign the
               dictionary to the absinfo code.
               This is different to the libevdev behavior.
        """
    if not isinstance(code, int):
        if not code.startswith('ABS_'):
            raise ValueError()
        code = self.event_to_value('EV_ABS', code)
    absinfo = self._get_abs_info(self._ctx, code)
    if not absinfo:
        return None
    if new_values is not None:
        if 'minimum' in new_values:
            absinfo.contents.minimum = new_values['minimum']
        if 'maximum' in new_values:
            absinfo.contents.maximum = new_values['maximum']
        if 'value' in new_values:
            absinfo.contents.value = new_values['value']
        if 'fuzz' in new_values:
            absinfo.contents.fuzz = new_values['fuzz']
        if 'flat' in new_values:
            absinfo.contents.flat = new_values['flat']
        if 'resolution' in new_values:
            absinfo.contents.resolution = new_values['resolution']
        if kernel:
            rc = self._kernel_set_abs_info(self._ctx, code, absinfo)
            if rc != 0:
                raise OSError(-rc, os.strerror(-rc))
    return {'value': absinfo.contents.value, 'minimum': absinfo.contents.minimum, 'maximum': absinfo.contents.maximum, 'fuzz': absinfo.contents.fuzz, 'flat': absinfo.contents.flat, 'resolution': absinfo.contents.resolution}