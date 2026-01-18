import ctypes
from pyglet.libs.win32 import com
class IDirectInput8(com.pIUnknown):
    _methods_ = [('CreateDevice', com.STDMETHOD(ctypes.POINTER(com.GUID), ctypes.POINTER(IDirectInputDevice8), ctypes.c_void_p)), ('EnumDevices', com.STDMETHOD(DWORD, LPDIENUMDEVICESCALLBACK, LPVOID, DWORD)), ('GetDeviceStatus', com.STDMETHOD()), ('RunControlPanel', com.STDMETHOD()), ('Initialize', com.STDMETHOD()), ('FindDevice', com.STDMETHOD()), ('EnumDevicesBySemantics', com.STDMETHOD()), ('ConfigureDevices', com.STDMETHOD())]