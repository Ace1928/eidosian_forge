from winappdbg.win32.defines import *
def GetLargePageMinimum():
    _GetLargePageMinimum = windll.user32.GetLargePageMinimum
    _GetLargePageMinimum.argtypes = []
    _GetLargePageMinimum.restype = SIZE_T
    return _GetLargePageMinimum()