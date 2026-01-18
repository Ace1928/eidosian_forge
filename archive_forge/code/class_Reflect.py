from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
class Reflect:

    def __init__(self, name: str, rval=None):
        self._name = name
        self._rval = rval

    def __call__(self, *argl, **argd):
        args = ', '.join([repr(a) for a in argl])
        if args and argd:
            args = f'{args}, '
        args += ', '.join([f'{k}={v!r}' for k, v in argd.items()])
        print(f'{self._name}({args})')
        if loop_exit:
            raise ExitMainLoop()
        return self._rval

    def __getattr__(self, attr):
        if attr.endswith('_rval'):
            raise AttributeError()
        if hasattr(self, f'{attr}_rval'):
            return Reflect(f'{self._name}.{attr}', getattr(self, f'{attr}_rval'))
        return Reflect(f'{self._name}.{attr}')