import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
from IPython.testing.tools import make_tempfile
from IPython.utils import path
class TestShellGlob(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filenames_start_with_a = ['a0', 'a1', 'a2']
        cls.filenames_end_with_b = ['0b', '1b', '2b']
        cls.filenames = cls.filenames_start_with_a + cls.filenames_end_with_b
        cls.tempdir = TemporaryDirectory()
        td = cls.tempdir.name
        with cls.in_tempdir():
            for fname in cls.filenames:
                open(os.path.join(td, fname), 'w', encoding='utf-8').close()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    @contextmanager
    def in_tempdir(cls):
        save = os.getcwd()
        try:
            os.chdir(cls.tempdir.name)
            yield
        finally:
            os.chdir(save)

    def check_match(self, patterns, matches):
        with self.in_tempdir():
            assert sorted(path.shellglob(patterns)) == sorted(matches)

    def common_cases(self):
        return [(['*'], self.filenames), (['a*'], self.filenames_start_with_a), (['*c'], ['*c']), (['*', 'a*', '*b', '*c'], self.filenames + self.filenames_start_with_a + self.filenames_end_with_b + ['*c']), (['a[012]'], self.filenames_start_with_a)]

    @skip_win32
    def test_match_posix(self):
        for patterns, matches in self.common_cases() + [(['\\*'], ['*']), (['a\\*', 'a*'], ['a*'] + self.filenames_start_with_a), (['a\\[012]'], ['a[012]'])]:
            self.check_match(patterns, matches)

    @skip_if_not_win32
    def test_match_windows(self):
        for patterns, matches in self.common_cases() + [(['a\\*', 'a*'], ['a\\*'] + self.filenames_start_with_a), (['a\\[012]'], ['a\\[012]'])]:
            self.check_match(patterns, matches)