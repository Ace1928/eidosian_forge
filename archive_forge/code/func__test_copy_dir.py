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
def _test_copy_dir(self, protocol):
    other_fs = open_fs(protocol)
    self.fs.makedirs('foo/bar/baz')
    self.fs.makedir('egg')
    self.fs.writetext('top.txt', 'Hello, World')
    self.fs.writetext('/foo/bar/baz/test.txt', 'Goodbye, World')
    fs.copy.copy_dir(self.fs, '/', other_fs, '/')
    expected = {'/egg', '/foo', '/foo/bar', '/foo/bar/baz'}
    self.assertEqual(set(walk.walk_dirs(other_fs)), expected)
    self.assert_text('top.txt', 'Hello, World')
    self.assert_text('/foo/bar/baz/test.txt', 'Goodbye, World')
    other_fs = open_fs('mem://')
    fs.copy.copy_dir(self.fs, '/foo', other_fs, '/')
    self.assertEqual(list(walk.walk_files(other_fs)), ['/bar/baz/test.txt'])
    print('BEFORE')
    self.fs.tree()
    other_fs.tree()
    fs.copy.copy_dir(self.fs, '/foo', other_fs, '/egg')
    print('FS')
    self.fs.tree()
    print('OTHER')
    other_fs.tree()
    self.assertEqual(list(walk.walk_files(other_fs)), ['/bar/baz/test.txt', '/egg/bar/baz/test.txt'])