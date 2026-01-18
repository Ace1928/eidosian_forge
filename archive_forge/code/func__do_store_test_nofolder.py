import unittest
from os.path import abspath, dirname, join
import errno
import os
def _do_store_test_nofolder(self, store_cls):
    ext = store_cls.__name__.lower()[:4]
    path = join(dirname(abspath(__file__)), '__i_dont_exist__', 'test.' + ext)
    with self.assertRaises(IOError) as context:
        store = store_cls(path)
    self.assertEqual(context.exception.errno, errno.ENOENT)