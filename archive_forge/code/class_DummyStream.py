from gitdb import OStream
import sys
import random
from array import array
from io import BytesIO
import glob
import unittest
import tempfile
import shutil
import os
import gc
import logging
from functools import wraps
class DummyStream:

    def __init__(self):
        self.was_read = False
        self.bytes = 0
        self.closed = False

    def read(self, size):
        self.was_read = True
        self.bytes = size

    def close(self):
        self.closed = True

    def _assert(self):
        assert self.was_read