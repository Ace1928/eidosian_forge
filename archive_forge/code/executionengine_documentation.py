import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file

        Low-level getbuffer hook.
        