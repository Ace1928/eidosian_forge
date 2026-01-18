from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
class SerialLiar(object):
    """Attribute accesses always get another copy of the same class.

    unittest.mock.call does something similar, but it's not ideal for testing
    as the failure mode is to eat all your RAM. This gives up after 10k levels.
    """

    def __init__(self, max_fibbing_twig, lies_told=0):
        if lies_told > 10000:
            raise RuntimeError('Nose too long, honesty is the best policy')
        self.max_fibbing_twig = max_fibbing_twig
        self.lies_told = lies_told
        max_fibbing_twig[0] = max(max_fibbing_twig[0], lies_told)

    def __getattr__(self, item):
        return SerialLiar(self.max_fibbing_twig, self.lies_told + 1)