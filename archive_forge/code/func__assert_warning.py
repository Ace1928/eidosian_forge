from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
def _assert_warning(self, warnings, cls):
    for w in warnings:
        if isinstance(w.message, cls):
            assert w.filename == __file__
            return w
    raise Exception('%s warning not found in %r' % (cls, warnings))