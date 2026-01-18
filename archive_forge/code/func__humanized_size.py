from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def _humanized_size(self, size):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E']:
        if size < 1024.0:
            return '%3.2f%sB' % (size, unit)
        size /= 1024.0
    return '%.2f%sB' % (size, 'Z')