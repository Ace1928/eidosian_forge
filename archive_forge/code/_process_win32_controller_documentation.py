import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
Creates a Windows pipe, which consists of two handles.

                The 'uninherit' parameter controls which handle is not
                inherited by the child process.
                