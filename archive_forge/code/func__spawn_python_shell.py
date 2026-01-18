from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.util import PathOperations
from winappdbg.event import EventHandler, NoEvent
from winappdbg.textio import HexInput, HexOutput, HexDump, CrashDump, DebugLog
import os
import sys
import code
import time
import warnings
import traceback
from cmd import Cmd
def _spawn_python_shell(self, arg):
    import winappdbg
    banner = 'Python %s on %s\nType "help", "copyright", "credits" or "license" for more information.\n'
    platform = winappdbg.version.lower()
    platform = 'WinAppDbg %s' % platform
    banner = banner % (sys.version, platform)
    local = {}
    local.update(__builtins__)
    local.update({'__name__': '__console__', '__doc__': None, 'exit': self._python_exit, 'self': self, 'arg': arg, 'winappdbg': winappdbg})
    try:
        code.interact(banner=banner, local=local)
    except SystemExit:
        pass