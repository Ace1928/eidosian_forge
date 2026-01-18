from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class _PrintableStructure(Structure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)

    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default fomratting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"

    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}

    def __str__(self):
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = '%s'
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif '<default>' in self._fmt_:
                fmt = self._fmt_['<default>']
            result.append(('%s: ' + fmt) % (key, value))
        return self.__class__.__name__ + '(' + ', '.join(result) + ')'

    def __getattribute__(self, name):
        res = super(_PrintableStructure, self).__getattribute__(name)
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res

    def __setattr__(self, name, value):
        if isinstance(value, str):
            value = value.encode()
        super(_PrintableStructure, self).__setattr__(name, value)