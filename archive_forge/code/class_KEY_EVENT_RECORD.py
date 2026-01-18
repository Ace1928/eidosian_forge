from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class KEY_EVENT_RECORD(Structure):
    """
    http://msdn.microsoft.com/en-us/library/windows/desktop/ms684166(v=vs.85).aspx
    """
    _fields_ = [('KeyDown', c_long), ('RepeatCount', c_short), ('VirtualKeyCode', c_short), ('VirtualScanCode', c_short), ('uChar', UNICODE_OR_ASCII), ('ControlKeyState', c_long)]