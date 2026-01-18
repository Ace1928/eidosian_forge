from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class FixIt(object):
    """
    A FixIt represents a transformation to be applied to the source to
    "fix-it". The fix-it shouldbe applied by replacing the given source range
    with the given value.
    """

    def __init__(self, range, value):
        self.range = range
        self.value = value

    def __repr__(self):
        return '<FixIt range %r, value %r>' % (self.range, self.value)