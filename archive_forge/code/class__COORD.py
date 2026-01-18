import ctypes
class _COORD(ctypes.Structure):
    _fields_ = [('X', SHORT), ('Y', SHORT)]