import os
import six
import sys
from ctypes import cdll
from ctypes import CFUNCTYPE
from ctypes import CDLL
from ctypes import POINTER
from ctypes import Structure
from ctypes import byref
from ctypes import cast
from ctypes import sizeof
from ctypes import py_object
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_size_t
from ctypes import c_void_p
from ctypes import memmove
from ctypes.util import find_library
from typing import Union
@conv_func
def __conv(n_messages, messages, p_response, app_data):
    pyob = cast(app_data, py_object).value
    msg_list = pyob.get('msgs')
    password = pyob.get('password')
    encoding = pyob.get('encoding')
    return my_conv(n_messages, messages, p_response, self.libc, msg_list, password, encoding)