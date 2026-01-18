from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class Availability(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<Availability: %s>' % self