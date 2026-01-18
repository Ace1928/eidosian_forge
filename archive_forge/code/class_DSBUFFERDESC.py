import ctypes
from pyglet.libs.win32 import com
class DSBUFFERDESC(ctypes.Structure):
    _fields_ = [('dwSize', DWORD), ('dwFlags', DWORD), ('dwBufferBytes', DWORD), ('dwReserved', DWORD), ('lpwfxFormat', LPWAVEFORMATEX)]

    def __repr__(self):
        return 'DSBUFFERDESC(dwSize={}, dwFlags={}, dwBufferBytes={}, lpwfxFormat={})'.format(self.dwSize, self.dwFlags, self.dwBufferBytes, self.lpwfxFormat.contents if self.lpwfxFormat else None)