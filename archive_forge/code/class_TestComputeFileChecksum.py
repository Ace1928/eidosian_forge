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
class TestComputeFileChecksum(test_base.BaseTestCase):

    def setUp(self):
        super(TestComputeFileChecksum, self).setUp()
        self.content = 'fake_content'.encode('ascii')

    def check_file_content(self, content, path):
        with open(path, 'r') as fd:
            ans = fd.read()
            self.assertEqual(content, ans.encode('latin-1'))

    def test_compute_checksum_default_algorithm(self):
        path = fileutils.write_to_tempfile(self.content)
        self.assertTrue(os.path.exists(path))
        self.check_file_content(self.content, path)
        expected_checksum = hashlib.sha256()
        expected_checksum.update(self.content)
        actual_checksum = fileutils.compute_file_checksum(path)
        self.assertEqual(expected_checksum.hexdigest(), actual_checksum)

    def test_compute_checksum_sleep_0_called(self):
        path = fileutils.write_to_tempfile(self.content)
        self.assertTrue(os.path.exists(path))
        self.check_file_content(self.content, path)
        expected_checksum = hashlib.sha256()
        expected_checksum.update(self.content)
        with mock.patch.object(time, 'sleep') as sleep_mock:
            actual_checksum = fileutils.compute_file_checksum(path, read_chunksize=4)
        sleep_mock.assert_has_calls([mock.call(0)] * 3)
        self.assertEqual(3, sleep_mock.call_count)
        self.assertEqual(expected_checksum.hexdigest(), actual_checksum)

    def test_compute_checksum_named_algorithm(self):
        path = fileutils.write_to_tempfile(self.content)
        self.assertTrue(os.path.exists(path))
        self.check_file_content(self.content, path)
        expected_checksum = hashlib.sha512()
        expected_checksum.update(self.content)
        actual_checksum = fileutils.compute_file_checksum(path, algorithm='sha512')
        self.assertEqual(expected_checksum.hexdigest(), actual_checksum)

    def test_compute_checksum_invalid_algorithm(self):
        path = fileutils.write_to_tempfile(self.content)
        self.assertTrue(os.path.exists(path))
        self.check_file_content(self.content, path)
        self.assertRaises(ValueError, fileutils.compute_file_checksum, path, algorithm='foo')

    def test_file_does_not_exist(self):
        random_file_name = uuid.uuid4().hex
        path = os.path.join('/tmp', random_file_name)
        self.assertRaises(IOError, fileutils.compute_file_checksum, path)

    def test_generic_io_error(self):
        tempdir = tempfile.mkdtemp()
        self.assertRaises(IOError, fileutils.compute_file_checksum, tempdir)