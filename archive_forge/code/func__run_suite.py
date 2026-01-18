from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def _run_suite(suite):
    """Run tests from a unittest.TestSuite-derived class."""
    if verbose:
        runner = unittest.TextTestRunner(sys.stdout, verbosity=2, failfast=failfast)
    else:
        runner = BasicTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        if len(result.errors) == 1 and (not result.failures):
            err = result.errors[0][1]
        elif len(result.failures) == 1 and (not result.errors):
            err = result.failures[0][1]
        else:
            err = 'multiple errors occurred'
            if not verbose:
                err += '; run in verbose mode for details'
        raise TestFailed(err)