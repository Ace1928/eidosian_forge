import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class _DAVStat:
    """The stat info as it can be acquired with DAV."""

    def __init__(self, size, is_dir, is_exec):
        self.st_size = size
        if is_dir:
            self.st_mode = 16804
        else:
            self.st_mode = 33188
        if is_exec:
            self.st_mode = self.st_mode | 493