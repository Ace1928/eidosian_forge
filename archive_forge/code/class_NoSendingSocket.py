import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class NoSendingSocket:

    def __init__(self):
        self.call_count = 0

    def send(self, bytes):
        self.call_count += 1
        if self.call_count > 100:
            raise RuntimeError('too many calls')
        return 0