from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class COORD(Structure):
    """
    Struct in wincon.h
    http://msdn.microsoft.com/en-us/library/windows/desktop/ms682119(v=vs.85).aspx
    """
    _fields_ = [('X', c_short), ('Y', c_short)]

    def __repr__(self):
        return '%s(X=%r, Y=%r, type_x=%r, type_y=%r)' % (self.__class__.__name__, self.X, self.Y, type(self.X), type(self.Y))