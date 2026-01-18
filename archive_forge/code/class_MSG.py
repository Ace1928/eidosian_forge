import ctypes
class MSG(ctypes.Structure):
    _fields_ = [('hWnd', HWND), ('message', UINT), ('wParam', WPARAM), ('lParam', LPARAM), ('time', DWORD), ('pt', POINT)]