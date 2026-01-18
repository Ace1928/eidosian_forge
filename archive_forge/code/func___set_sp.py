from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
def __set_sp(self, value):
    self['Esp'] = value