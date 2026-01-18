import os
from threading import Lock
import warnings
from .mmap_dict import mmap_key, MmapedDict
def __check_for_pid_change(self):
    actual_pid = process_identifier()
    if pid['value'] != actual_pid:
        pid['value'] = actual_pid
        for f in files.values():
            f.close()
        files.clear()
        for value in values:
            value.__reset()