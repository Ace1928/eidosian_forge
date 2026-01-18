from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class Kind(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<ChunkKind: %s>' % self