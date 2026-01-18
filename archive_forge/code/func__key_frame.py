from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def _key_frame(self, parent, stackframe):
    return (parent, stackframe.filename, stackframe.lineno, stackframe.name)