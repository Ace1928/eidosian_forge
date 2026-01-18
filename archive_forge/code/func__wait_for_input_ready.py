from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def _wait_for_input_ready(self, timeout: float | None) -> list[int]:
    logger = self.logger.getChild('wait_for_input_ready')
    fd_list = self.get_input_descriptors()
    logger.debug(f'Waiting for input: descriptors={fd_list!r}, timeout={timeout!r}')
    with selectors.DefaultSelector() as selector:
        for fd in fd_list:
            selector.register(fd, selectors.EVENT_READ)
        ready = selector.select(timeout)
    logger.debug(f'Input ready: {ready}')
    return [event.fd for event, _ in ready]