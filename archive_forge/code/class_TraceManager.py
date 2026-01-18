import codecs
import contextlib
import locale
import logging
import math
import os
from functools import partial
from typing import TextIO, Union
import dill
class TraceManager(contextlib.AbstractContextManager):
    """context manager version of trace(); can redirect the trace to a file"""

    def __init__(self, file, mode):
        self.file = file
        self.mode = mode
        self.redirect = file is not None
        self.file_is_stream = hasattr(file, 'write')

    def __enter__(self):
        if self.redirect:
            stderr_handler.flush()
            if self.file_is_stream:
                self.handler = logging.StreamHandler(self.file)
            else:
                self.handler = logging.FileHandler(self.file, self.mode)
            adapter.removeHandler(stderr_handler)
            adapter.addHandler(self.handler)
        self.old_level = adapter.getEffectiveLevel()
        adapter.setLevel(logging.INFO)
        return adapter.info

    def __exit__(self, *exc_info):
        adapter.setLevel(self.old_level)
        if self.redirect:
            adapter.removeHandler(self.handler)
            adapter.addHandler(stderr_handler)
            if not self.file_is_stream:
                self.handler.close()