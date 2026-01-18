from __future__ import division
import array
import os
import subprocess
from tempfile import TemporaryFile, NamedTemporaryFile
import wave
import sys
import struct
from .logging_utils import log_conversion, log_subprocess_output
from .utils import mediainfo_json, fsdecode
import base64
from collections import namedtuple
from io import BytesIO
from .utils import (
from .exceptions import (
from . import effects
class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self