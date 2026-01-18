import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def add_current_process(self):
    """
        Allows the JITted library to access symbols in the current binary.

        That is, it allows exporting the current binary's symbols, including
        loaded libraries, as imports to the JITted
        library.
        """
    self.__entries.append((3, b''))
    return self