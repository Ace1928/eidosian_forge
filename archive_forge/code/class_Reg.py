import os
import subprocess
import sys
import re
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
import winreg
class Reg:
    """Helper class to read values from the registry
    """

    def get_value(cls, path, key):
        for base in HKEYS:
            d = cls.read_values(base, path)
            if d and key in d:
                return d[key]
        raise KeyError(key)
    get_value = classmethod(get_value)

    def read_keys(cls, base, key):
        """Return list of registry keys."""
        try:
            handle = RegOpenKeyEx(base, key)
        except RegError:
            return None
        L = []
        i = 0
        while True:
            try:
                k = RegEnumKey(handle, i)
            except RegError:
                break
            L.append(k)
            i += 1
        return L
    read_keys = classmethod(read_keys)

    def read_values(cls, base, key):
        """Return dict of registry keys and values.

        All names are converted to lowercase.
        """
        try:
            handle = RegOpenKeyEx(base, key)
        except RegError:
            return None
        d = {}
        i = 0
        while True:
            try:
                name, value, type = RegEnumValue(handle, i)
            except RegError:
                break
            name = name.lower()
            d[cls.convert_mbcs(name)] = cls.convert_mbcs(value)
            i += 1
        return d
    read_values = classmethod(read_values)

    def convert_mbcs(s):
        dec = getattr(s, 'decode', None)
        if dec is not None:
            try:
                s = dec('mbcs')
            except UnicodeError:
                pass
        return s
    convert_mbcs = staticmethod(convert_mbcs)