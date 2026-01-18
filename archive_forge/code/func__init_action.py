from __future__ import annotations
import os
from typing import Any, Callable
def _init_action(self, action):
    from ctypes import WINFUNCTYPE, windll
    from ctypes.wintypes import BOOL, DWORD
    kernel32 = windll.LoadLibrary('kernel32')
    PHANDLER_ROUTINE = WINFUNCTYPE(BOOL, DWORD)
    SetConsoleCtrlHandler = self._SetConsoleCtrlHandler = kernel32.SetConsoleCtrlHandler
    SetConsoleCtrlHandler.argtypes = (PHANDLER_ROUTINE, BOOL)
    SetConsoleCtrlHandler.restype = BOOL
    if action is None:

        def action():
            return None
    self.action = action

    @PHANDLER_ROUTINE
    def handle(event):
        if event == 0:
            action()
        return 0
    self.handle = handle