from __future__ import (absolute_import, division, print_function)
import os
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ctypes import CDLL, c_char_p, c_int, byref, POINTER, get_errno
class _to_char_p:

    @classmethod
    def from_param(cls, strvalue):
        if strvalue is not None and (not isinstance(strvalue, binary_char_type)):
            strvalue = to_bytes(strvalue)
        return strvalue