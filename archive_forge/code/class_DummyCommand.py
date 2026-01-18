import os
import sys
import shutil
import tempfile
import unittest
import sysconfig
from copy import deepcopy
from test.support import os_helper
from distutils import log
from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL
from distutils.core import Distribution
class DummyCommand:
    """Class to store options for retrieval via set_undefined_options()."""

    def __init__(self, **kwargs):
        for kw, val in kwargs.items():
            setattr(self, kw, val)

    def ensure_finalized(self):
        pass