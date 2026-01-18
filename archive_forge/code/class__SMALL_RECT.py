import ctypes
class _SMALL_RECT(ctypes.Structure):
    _fields_ = [('Left', SHORT), ('Top', SHORT), ('Right', SHORT), ('Bottom', SHORT)]