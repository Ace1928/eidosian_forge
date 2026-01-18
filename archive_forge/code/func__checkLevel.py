import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def _checkLevel(level):
    if isinstance(level, int):
        rv = level
    elif str(level) == level:
        if level not in _nameToLevel:
            raise ValueError('Unknown level: %r' % level)
        rv = _nameToLevel[level]
    else:
        raise TypeError('Level not an integer or a valid string: %r' % (level,))
    return rv