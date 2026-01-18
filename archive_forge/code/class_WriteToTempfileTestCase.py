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
class WriteToTempfileTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(WriteToTempfileTestCase, self).setUp()
        self.content = 'testing123'.encode('ascii')

    def check_file_content(self, path):
        with open(path, 'r') as fd:
            ans = fd.read()
            self.assertEqual(self.content, ans.encode('latin-1'))

    def test_file_without_path_and_suffix(self):
        res = fileutils.write_to_tempfile(self.content)
        self.assertTrue(os.path.exists(res))
        basepath, tmpfile = os.path.split(res)
        self.assertTrue(basepath.startswith(tempfile.gettempdir()))
        self.assertTrue(tmpfile.startswith('tmp'))
        self.check_file_content(res)

    def test_file_with_not_existing_path(self):
        random_dir = uuid.uuid4().hex
        path = '/tmp/%s/test1' % random_dir
        res = fileutils.write_to_tempfile(self.content, path=path)
        self.assertTrue(os.path.exists(res))
        basepath, tmpfile = os.path.split(res)
        self.assertEqual(basepath, path)
        self.assertTrue(tmpfile.startswith('tmp'))
        self.check_file_content(res)
        shutil.rmtree('/tmp/' + random_dir)

    def test_file_with_not_default_suffix(self):
        suffix = '.conf'
        res = fileutils.write_to_tempfile(self.content, suffix=suffix)
        self.assertTrue(os.path.exists(res))
        basepath, tmpfile = os.path.split(res)
        self.assertTrue(basepath.startswith(tempfile.gettempdir()))
        self.assertTrue(tmpfile.startswith('tmp'))
        self.assertTrue(tmpfile.endswith('.conf'))
        self.check_file_content(res)

    def test_file_with_not_existing_path_and_not_default_suffix(self):
        suffix = '.txt'
        random_dir = uuid.uuid4().hex
        path = '/tmp/%s/test2' % random_dir
        res = fileutils.write_to_tempfile(self.content, path=path, suffix=suffix)
        self.assertTrue(os.path.exists(res))
        basepath, tmpfile = os.path.split(res)
        self.assertTrue(tmpfile.startswith('tmp'))
        self.assertEqual(basepath, path)
        self.assertTrue(tmpfile.endswith(suffix))
        self.check_file_content(res)
        shutil.rmtree('/tmp/' + random_dir)

    def test_file_with_not_default_prefix(self):
        prefix = 'test'
        res = fileutils.write_to_tempfile(self.content, prefix=prefix)
        self.assertTrue(os.path.exists(res))
        basepath, tmpfile = os.path.split(res)
        self.assertTrue(tmpfile.startswith(prefix))
        self.assertTrue(basepath.startswith(tempfile.gettempdir()))
        self.check_file_content(res)