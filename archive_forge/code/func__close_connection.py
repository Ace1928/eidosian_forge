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
def _close_connection(self) -> None:
    if self.update_method == 'polling child':
        self.server_socket.settimeout(0)
        sock, _addr = self.server_socket.accept()
        sock.sendall(b'Z')
        sock.close()
    if self.update_method == 'multipart':
        sys.stdout.write('\r\nZ\r\n--ZZ--\r\n')
        sys.stdout.flush()