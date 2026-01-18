from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class MOUSE_EVENT_RECORD(Structure):
    """
    http://msdn.microsoft.com/en-us/library/windows/desktop/ms684239(v=vs.85).aspx
    """
    _fields_ = [('MousePosition', COORD), ('ButtonState', c_long), ('ControlKeyState', c_long), ('EventFlags', c_long)]