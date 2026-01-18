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
class TestLinkOrCopy(unittest.TestCase):

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.src = self.dst('src')
        with open(self.src, 'w', encoding='utf-8') as f:
            f.write('Hello, world!')

    def tearDown(self):
        self.tempdir.cleanup()

    def dst(self, *args):
        return os.path.join(self.tempdir.name, *args)

    def assert_inode_not_equal(self, a, b):
        assert os.stat(a).st_ino != os.stat(b).st_ino, '%r and %r do reference the same indoes' % (a, b)

    def assert_inode_equal(self, a, b):
        assert os.stat(a).st_ino == os.stat(b).st_ino, '%r and %r do not reference the same indoes' % (a, b)

    def assert_content_equal(self, a, b):
        with open(a, 'rb') as a_f:
            with open(b, 'rb') as b_f:
                assert a_f.read() == b_f.read()

    @skip_win32
    def test_link_successful(self):
        dst = self.dst('target')
        path.link_or_copy(self.src, dst)
        self.assert_inode_equal(self.src, dst)

    @skip_win32
    def test_link_into_dir(self):
        dst = self.dst('some_dir')
        os.mkdir(dst)
        path.link_or_copy(self.src, dst)
        expected_dst = self.dst('some_dir', os.path.basename(self.src))
        self.assert_inode_equal(self.src, expected_dst)

    @skip_win32
    def test_target_exists(self):
        dst = self.dst('target')
        open(dst, 'w', encoding='utf-8').close()
        path.link_or_copy(self.src, dst)
        self.assert_inode_equal(self.src, dst)

    @skip_win32
    def test_no_link(self):
        real_link = os.link
        try:
            del os.link
            dst = self.dst('target')
            path.link_or_copy(self.src, dst)
            self.assert_content_equal(self.src, dst)
            self.assert_inode_not_equal(self.src, dst)
        finally:
            os.link = real_link

    @skip_if_not_win32
    def test_windows(self):
        dst = self.dst('target')
        path.link_or_copy(self.src, dst)
        self.assert_content_equal(self.src, dst)

    def test_link_twice(self):
        dst = self.dst('target')
        path.link_or_copy(self.src, dst)
        path.link_or_copy(self.src, dst)
        self.assert_inode_equal(self.src, dst)
        assert sorted(os.listdir(self.tempdir.name)) == ['src', 'target']