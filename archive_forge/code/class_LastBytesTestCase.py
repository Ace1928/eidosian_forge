import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from unittest import mock
import uuid
import yaml
from oslotest import base as test_base
from oslo_utils import fileutils
class LastBytesTestCase(test_base.BaseTestCase):
    """Test the last_bytes() utility method."""

    def setUp(self):
        super(LastBytesTestCase, self).setUp()
        self.content = b'1234567890'

    def test_truncated(self):
        res = fileutils.write_to_tempfile(self.content)
        self.assertTrue(os.path.exists(res))
        out, unread_bytes = fileutils.last_bytes(res, 5)
        self.assertEqual(b'67890', out)
        self.assertGreater(unread_bytes, 0)

    def test_read_all(self):
        res = fileutils.write_to_tempfile(self.content)
        self.assertTrue(os.path.exists(res))
        out, unread_bytes = fileutils.last_bytes(res, 1000)
        self.assertEqual(b'1234567890', out)
        self.assertEqual(0, unread_bytes)

    def test_non_exist_file(self):
        self.assertRaises(IOError, fileutils.last_bytes, 'non_exist_file', 1000)