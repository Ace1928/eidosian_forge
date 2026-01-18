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
def _test_download(self, workers):
    """Test fs.copy with varying number of worker threads."""
    src_fs = self.fs
    with open_fs('temp://') as dst_fs:
        src_fs.writebytes('foo', self.data1)
        src_fs.writebytes('bar', self.data2)
        src_fs.makedir('dir1').writebytes('baz', self.data3)
        src_fs.makedirs('dir2/dir3').writebytes('egg', self.data4)
        fs.copy.copy_fs(src_fs, dst_fs, workers=workers)
        self.assertEqual(dst_fs.readbytes('foo'), self.data1)
        self.assertEqual(dst_fs.readbytes('bar'), self.data2)
        self.assertEqual(dst_fs.readbytes('dir1/baz'), self.data3)
        self.assertEqual(dst_fs.readbytes('dir2/dir3/egg'), self.data4)