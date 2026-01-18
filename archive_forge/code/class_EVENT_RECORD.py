from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class EVENT_RECORD(Union):
    _fields_ = [('KeyEvent', KEY_EVENT_RECORD), ('MouseEvent', MOUSE_EVENT_RECORD), ('WindowBufferSizeEvent', WINDOW_BUFFER_SIZE_RECORD), ('MenuEvent', MENU_EVENT_RECORD), ('FocusEvent', FOCUS_EVENT_RECORD)]