from __future__ import annotations
import contextlib
import functools
import logging
import selectors
import socket
import sys
import threading
import typing
from ctypes import byref
from ctypes.wintypes import DWORD
from urwid import signals
from . import _raw_display_base, _win32, escape
from .common import INPUT_DESCRIPTORS_CHANGED
class ReadInputThread(threading.Thread):
    name = 'urwid Windows input reader'
    daemon = True
    should_exit: bool = False

    def __init__(self, input_socket: socket.socket, resize: Callable[[], typing.Any]) -> None:
        self._input = input_socket
        self._resize = resize
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        super().__init__()

    def run(self) -> None:
        hIn = _win32.GetStdHandle(_win32.STD_INPUT_HANDLE)
        MAX = 2048
        read = DWORD(0)
        arrtype = _win32.INPUT_RECORD * MAX
        input_records = arrtype()
        while True:
            _win32.ReadConsoleInputW(hIn, byref(input_records), MAX, byref(read))
            if self.should_exit:
                return
            for i in range(read.value):
                inp = input_records[i]
                if inp.EventType == _win32.EventType.KEY_EVENT:
                    if not inp.Event.KeyEvent.bKeyDown:
                        continue
                    input_data = inp.Event.KeyEvent.uChar.AsciiChar
                    if input_data != b'\x00':
                        self._input.send(input_data)
                elif inp.EventType == _win32.EventType.WINDOW_BUFFER_SIZE_EVENT:
                    self._resize()
                else:
                    pass