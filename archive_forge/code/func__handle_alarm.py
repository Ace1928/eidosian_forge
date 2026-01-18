from __future__ import annotations
import dataclasses
import glob
import html
import os
import pathlib
import random
import selectors
import signal
import socket
import string
import sys
import tempfile
import typing
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width, move_next_char
from urwid.util import StoppingContext, get_encoding
from .common import BaseScreen
def _handle_alarm(self, sig, frame) -> None:
    if self.update_method not in {'multipart', 'polling child'}:
        raise ValueError(self.update_method)
    if self.update_method == 'polling child':
        try:
            s, _addr = self.server_socket.accept()
            s.close()
        except socket.timeout:
            sys.exit(0)
    else:
        sys.stdout.write('\r\n\r\n--ZZ\r\n')
        sys.stdout.flush()
    signal.alarm(ALARM_DELAY)