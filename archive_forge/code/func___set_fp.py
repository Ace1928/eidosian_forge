from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
def __set_fp(self, value):
    self['Ebp'] = value