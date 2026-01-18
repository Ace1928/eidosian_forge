import sys
import os
def _get_win_folder_with_ctypes(csidl_name):
    import ctypes
    csidl_const = {'CSIDL_APPDATA': 26, 'CSIDL_COMMON_APPDATA': 35, 'CSIDL_LOCAL_APPDATA': 28}[csidl_name]
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)
    has_high_char = False
    for c in buf:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf2 = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2
    return buf.value