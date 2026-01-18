import unittest
from ctypes import *
import re, sys
class BEPoint(BigEndianStructure):
    _fields_ = [('x', c_long), ('y', c_long)]