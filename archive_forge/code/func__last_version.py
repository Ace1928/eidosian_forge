import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def _last_version(libnames, sep):

    def _num_version(libname):
        parts = libname.split(sep)
        nums = []
        try:
            while parts:
                nums.insert(0, int(parts.pop()))
        except ValueError:
            pass
        return nums or [maxsize]
    return max(reversed(libnames), key=_num_version)