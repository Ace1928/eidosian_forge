import contextlib
import hashlib
import os
import time
import unittest
from gzip import GzipFile
from io import BytesIO, UnsupportedOperation
from unittest import mock
import pytest
from packaging.version import Version
from ..deprecator import ExpiredDeprecationError
from ..openers import HAVE_INDEXED_GZIP, BZ2File, DeterministicGzipFile, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
class Lunk:
    closed = False

    def __init__(self, message):
        self.message = message

    def write(self):
        pass

    def read(self, size=-1, /):
        return self.message