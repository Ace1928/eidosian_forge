import pythran.config as cfg
from collections import defaultdict
import os.path
import os
def _find_exe(exe, *args, **kwargs):
    if exe == 'cl.exe':
        exe = ext.cc
    return find_exe(exe, *args, **kwargs)