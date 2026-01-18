from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class LibclangError(Exception):

    def __init__(self, message):
        self.m = message

    def __str__(self):
        return self.m