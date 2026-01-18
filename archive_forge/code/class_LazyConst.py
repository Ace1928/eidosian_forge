from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
class LazyConst(object):
    """Class whose attributes are computed on first access from its methods."""

    def __init__(self, cls):
        self._cls = cls
        self.__doc__ = getattr(cls, '__doc__')

    def __getattr__(self, name):
        func = getattr(self._cls, name)
        if not callable(func):
            return func
        try:
            value = func()
        except TypeError:
            value = func.__func__()
        setattr(self, name, value)
        return value