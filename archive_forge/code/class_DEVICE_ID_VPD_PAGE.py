import ctypes
import os
import re
import threading
from collections.abc import Iterable
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import pathutils
from os_win.utils import win32utils
from os_win.utils.winapi import libs as w_lib
class DEVICE_ID_VPD_PAGE(ctypes.BigEndianStructure):
    _fields_ = [('DeviceType', ctypes.c_ubyte, 5), ('Qualifier', ctypes.c_ubyte, 3), ('PageCode', ctypes.c_ubyte), ('PageLength', ctypes.c_uint16)]