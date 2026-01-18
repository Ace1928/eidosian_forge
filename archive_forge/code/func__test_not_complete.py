import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def _test_not_complete(reason, s, comp):
    l = len(s)
    with provisionalcompleter():
        ip.Completer.use_jedi = True
        completions = set(ip.Completer.completions(s, l))
        ip.Completer.use_jedi = False
        assert Completion(l, l, comp) not in completions, reason