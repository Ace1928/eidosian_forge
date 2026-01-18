import unittest
from ctypes import *
class BrokenStructure(Structure):

    def __init_subclass__(cls, **kwargs):
        cls._fields_ = []