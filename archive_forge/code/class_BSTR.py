import unittest
import sys
from ctypes import *
class BSTR(_SimpleCData):
    _type_ = 'X'