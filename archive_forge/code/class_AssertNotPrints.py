import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from io import StringIO
from subprocess import Popen, PIPE
from unittest.mock import patch
from traitlets.config.loader import Config
from IPython.utils.process import get_output_error_code
from IPython.utils.text import list_strings
from IPython.utils.io import temp_pyfile, Tee
from IPython.utils import py3compat
from . import decorators as dec
from . import skipdoctest
class AssertNotPrints(AssertPrints):
    """Context manager for checking that certain output *isn't* produced.

    Counterpart of AssertPrints"""

    def __exit__(self, etype, value, traceback):
        __tracebackhide__ = True
        try:
            if value is not None:
                self.tee.close()
                return False
            self.tee.flush()
            setattr(sys, self.channel, self.orig_stream)
            printed = self.buffer.getvalue()
            for s in self.s:
                if isinstance(s, _re_type):
                    assert not s.search(printed), printed_msg.format(s.pattern, self.channel, printed)
                else:
                    assert s not in printed, printed_msg.format(s, self.channel, printed)
            return False
        finally:
            self.tee.close()