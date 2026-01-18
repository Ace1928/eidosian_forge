import os
import unittest
import pytest
from monty.io import (
class TestFileLock:

    def setup_method(self):
        self.file_name = '__lock__'
        self.lock = FileLock(self.file_name, timeout=1)
        self.lock.acquire()

    def test_raise(self):
        with pytest.raises(FileLockException):
            new_lock = FileLock(self.file_name, timeout=1)
            new_lock.acquire()

    def teardown_method(self):
        self.lock.release()