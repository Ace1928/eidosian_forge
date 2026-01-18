from __future__ import absolute_import, unicode_literals
import io
import itertools
import json
import os
import six
import time
import unittest
import warnings
from datetime import datetime
from six import text_type
import fs.copy
import fs.move
from fs import ResourceType, Seek, errors, glob, walk
from fs.opener import open_fs
from fs.subfs import ClosingSubFS, SubFS
def _test_copy_dir_write(self, protocol):
    other_fs = open_fs(protocol)
    other_fs.makedirs('foo/bar/baz')
    other_fs.makedir('egg')
    other_fs.writetext('top.txt', 'Hello, World')
    other_fs.writetext('/foo/bar/baz/test.txt', 'Goodbye, World')
    fs.copy.copy_dir(other_fs, '/', self.fs, '/')
    expected = {'/egg', '/foo', '/foo/bar', '/foo/bar/baz'}
    self.assertEqual(set(walk.walk_dirs(self.fs)), expected)
    self.assert_text('top.txt', 'Hello, World')
    self.assert_text('/foo/bar/baz/test.txt', 'Goodbye, World')