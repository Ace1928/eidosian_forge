import unittest
from ctypes import *
import re, sys
class LEPoint(LittleEndianStructure):
    _fields_ = [('x', c_long), ('y', c_long)]