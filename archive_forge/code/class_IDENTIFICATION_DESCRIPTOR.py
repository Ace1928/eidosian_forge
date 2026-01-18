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
class IDENTIFICATION_DESCRIPTOR(ctypes.Structure):
    _fields_ = [('CodeSet', ctypes.c_ubyte, 4), ('ProtocolIdentifier', ctypes.c_ubyte, 4), ('IdentifierType', ctypes.c_ubyte, 4), ('Association', ctypes.c_ubyte, 2), ('_reserved', ctypes.c_ubyte, 1), ('Piv', ctypes.c_ubyte, 1), ('_reserved', ctypes.c_ubyte), ('IdentifierLength', ctypes.c_ubyte)]