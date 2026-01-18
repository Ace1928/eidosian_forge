import contextlib
import os
import shutil
import sys
import tempfile
import unittest
import six
@contextlib.contextmanager
def TempDir(change_to=False):
    if change_to:
        original_dir = os.getcwd()
    path = tempfile.mkdtemp()
    try:
        if change_to:
            os.chdir(path)
        yield path
    finally:
        if change_to:
            os.chdir(original_dir)
        shutil.rmtree(path)