from contextlib import contextmanager
import glob
from importlib import import_module
import io
import itertools
import os.path as osp
import re
import sys
import warnings
import zipfile
import configparser
class BadEntryPoint(Exception):
    """Raised when an entry point can't be parsed.
    """

    def __init__(self, epstr):
        self.epstr = epstr

    def __str__(self):
        return "Couldn't parse entry point spec: %r" % self.epstr

    @staticmethod
    @contextmanager
    def err_to_warnings():
        try:
            yield
        except BadEntryPoint as e:
            warnings.warn(str(e))