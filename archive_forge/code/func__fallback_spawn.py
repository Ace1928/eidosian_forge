import os
import subprocess
import contextlib
import warnings
import unittest.mock as mock
from .errors import (
from .ccompiler import CCompiler, gen_lib_options
from ._log import log
from .util import get_platform
from itertools import count
@contextlib.contextmanager
def _fallback_spawn(self, cmd, env):
    """
        Discovered in pypa/distutils#15, some tools monkeypatch the compiler,
        so the 'env' kwarg causes a TypeError. Detect this condition and
        restore the legacy, unsafe behavior.
        """
    bag = type('Bag', (), {})()
    try:
        yield bag
    except TypeError as exc:
        if "unexpected keyword argument 'env'" not in str(exc):
            raise
    else:
        return
    warnings.warn('Fallback spawn triggered. Please update distutils monkeypatch.')
    with mock.patch.dict('os.environ', env):
        bag.value = super().spawn(cmd)