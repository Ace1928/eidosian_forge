import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class TransformMark(str):
    """A mark on a transformation stream."""
    __slots__ = []
    _instances = {}

    def __new__(cls, val):
        return cls._instances.setdefault(val, str.__new__(cls, val))