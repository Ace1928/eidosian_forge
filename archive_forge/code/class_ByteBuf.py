import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
class ByteBuf:
    """
    Used by StdSim to write binary data and stores the actual bytes written
    """
    NEWLINES = [b'\n', b'\r']

    def __init__(self, std_sim_instance: StdSim) -> None:
        self.byte_buf = bytearray()
        self.std_sim_instance = std_sim_instance

    def write(self, b: bytes) -> None:
        """Add bytes to internal bytes buffer and if echo is True, echo contents to inner stream."""
        if not isinstance(b, bytes):
            raise TypeError(f'a bytes-like object is required, not {type(b)}')
        if not self.std_sim_instance.pause_storage:
            self.byte_buf += b
        if self.std_sim_instance.echo:
            self.std_sim_instance.inner_stream.buffer.write(b)
            if self.std_sim_instance.line_buffering:
                if any((newline in b for newline in ByteBuf.NEWLINES)):
                    self.std_sim_instance.flush()