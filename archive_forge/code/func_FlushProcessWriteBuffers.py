import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def FlushProcessWriteBuffers():
    _FlushProcessWriteBuffers = windll.kernel32.FlushProcessWriteBuffers
    _FlushProcessWriteBuffers.argtypes = []
    _FlushProcessWriteBuffers.restype = None
    _FlushProcessWriteBuffers()