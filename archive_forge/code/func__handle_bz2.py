import logging
import os.path
def _handle_bz2(file_obj, mode):
    from bz2 import BZ2File
    result = BZ2File(file_obj, mode)
    tweak_close(result, file_obj)
    return result