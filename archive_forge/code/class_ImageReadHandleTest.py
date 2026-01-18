import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class ImageReadHandleTest(base.TestCase):
    """Tests for ImageReadHandle."""

    def test_read(self):
        max_items = 10
        item = [1] * 10

        class ImageReadIterator(object):

            def __init__(self):
                self.num_items = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.num_items < max_items:
                    self.num_items += 1
                    return item
                raise StopIteration
            next = __next__
        handle = rw_handles.ImageReadHandle(ImageReadIterator())
        for _ in range(0, max_items):
            self.assertEqual(item, handle.read(10))
        self.assertFalse(handle.read(10))