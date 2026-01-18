import unittest
from ctypes import *
import _ctypes_test
class WNDCLASS(Structure):
    _fields_ = [('style', c_uint), ('lpfnWndProc', WNDPROC), ('cbClsExtra', c_int), ('cbWndExtra', c_int), ('hInstance', HINSTANCE), ('hIcon', HICON), ('hCursor', HCURSOR), ('lpszMenuName', LPCTSTR), ('lpszClassName', LPCTSTR)]