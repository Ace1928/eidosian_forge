from contextlib import contextmanager
from io import StringIO
import sys
import os
class StreamNonTTY(StringIO):

    def isatty(self):
        return False