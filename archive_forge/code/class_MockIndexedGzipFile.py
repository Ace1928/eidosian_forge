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
class MockIndexedGzipFile(GzipFile):

    def __init__(self, *args, **kwargs):
        self._drop_handles = kwargs.pop('drop_handles', False)
        super().__init__(*args, **kwargs)