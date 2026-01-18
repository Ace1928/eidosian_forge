import atexit
import ctypes
import os
import signal
import struct
import subprocess
import sys
import threading
from debugpy import launcher
from debugpy.common import log, messaging
from debugpy.launcher import output
def _wait_for_user_input():
    if sys.stdout and sys.stdin and sys.stdin.isatty():
        from debugpy.common import log
        try:
            import msvcrt
        except ImportError:
            can_getch = False
        else:
            can_getch = True
        if can_getch:
            log.debug('msvcrt available - waiting for user input via getch()')
            sys.stdout.write('Press any key to continue . . . ')
            sys.stdout.flush()
            msvcrt.getch()
        else:
            log.debug('msvcrt not available - waiting for user input via read()')
            sys.stdout.write('Press Enter to continue . . . ')
            sys.stdout.flush()
            sys.stdin.read(1)