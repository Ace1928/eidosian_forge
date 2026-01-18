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
class AssertPrints(object):
    """Context manager for testing that code prints certain text.

    Examples
    --------
    >>> with AssertPrints("abc", suppress=False):
    ...     print("abcd")
    ...     print("def")
    ...
    abcd
    def
    """

    def __init__(self, s, channel='stdout', suppress=True):
        self.s = s
        if isinstance(self.s, (str, _re_type)):
            self.s = [self.s]
        self.channel = channel
        self.suppress = suppress

    def __enter__(self):
        self.orig_stream = getattr(sys, self.channel)
        self.buffer = MyStringIO()
        self.tee = Tee(self.buffer, channel=self.channel)
        setattr(sys, self.channel, self.buffer if self.suppress else self.tee)

    def __exit__(self, etype, value, traceback):
        __tracebackhide__ = True
        try:
            if value is not None:
                return False
            self.tee.flush()
            setattr(sys, self.channel, self.orig_stream)
            printed = self.buffer.getvalue()
            for s in self.s:
                if isinstance(s, _re_type):
                    assert s.search(printed), notprinted_msg.format(s.pattern, self.channel, printed)
                else:
                    assert s in printed, notprinted_msg.format(s, self.channel, printed)
            return False
        finally:
            self.tee.close()