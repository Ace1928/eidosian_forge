import array
import ctypes.wintypes
import platform
import struct
from paramiko.common import zero_byte
from paramiko.util import b
import _thread as thread
from . import _winapi
def _query_pageant(msg):
    """
    Communication with the Pageant process is done through a shared
    memory-mapped file.
    """
    hwnd = _get_pageant_window_object()
    if not hwnd:
        return None
    map_name = f'PageantRequest{thread.get_ident():08x}'
    pymap = _winapi.MemoryMap(map_name, _AGENT_MAX_MSGLEN, _winapi.get_security_attributes_for_user())
    with pymap:
        pymap.write(msg)
        char_buffer = array.array('b', b(map_name) + zero_byte)
        char_buffer_address, char_buffer_size = char_buffer.buffer_info()
        cds = COPYDATASTRUCT(_AGENT_COPYDATA_ID, char_buffer_size, char_buffer_address)
        response = ctypes.windll.user32.SendMessageA(hwnd, win32con_WM_COPYDATA, ctypes.sizeof(cds), ctypes.byref(cds))
        if response > 0:
            pymap.seek(0)
            datalen = pymap.read(4)
            retlen = struct.unpack('>I', datalen)[0]
            return datalen + pymap.read(retlen)
        return None