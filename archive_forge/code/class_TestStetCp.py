from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import stat
from gslib import storage_url
from gslib.tests import testcase
from gslib.tests import util
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils import temporary_file_util
@unittest.skipIf(not system_util.IS_LINUX, 'STET binary supports only Linux.')
class TestStetCp(testcase.GsUtilIntegrationTestCase):
    """Integration tests for cp integration with STET binary."""

    def setUp(self):
        super(TestStetCp, self).setUp()
        self.stet_binary_path = self.CreateTempFile(contents=FAKE_STET_BINARY)
        current_stat = os.stat(self.stet_binary_path)
        os.chmod(self.stet_binary_path, current_stat.st_mode | stat.S_IEXEC)
        self.stet_config_path = self.CreateTempFile()

    def test_encrypts_upload_if_stet_is_enabled(self):
        object_uri = self.CreateObject()
        test_file = self.CreateTempFile(contents='will be rewritten')
        stderr = self.RunGsUtil(['-o', 'GSUtil:stet_binary_path={}'.format(self.stet_binary_path), '-o', 'GSUtil:stet_config_path={}'.format(self.stet_config_path), 'cp', '--stet', test_file, suri(object_uri)], return_stderr=True)
        self.assertNotIn('/4.0 B]', stderr)
        stdout = self.RunGsUtil(['cat', suri(object_uri)], return_stdout=True)
        self.assertIn('subcommand: encrypt', stdout)
        self.assertIn('config file: --config-file={}'.format(self.stet_config_path), stdout)
        self.assertIn('blob id: --blob-id={}'.format(suri(object_uri)), stdout)
        self.assertIn('in file: {}'.format(test_file), stdout)
        self.assertIn('out file: {}_.stet_tmp'.format(test_file), stdout)
        self.assertFalse(os.path.exists(temporary_file_util.GetStetTempFileName(storage_url.StorageUrlFromString(test_file))))

    def test_decrypts_download_if_stet_is_enabled(self):
        object_uri = self.CreateObject(contents='abc')
        test_file = self.CreateTempFile()
        stderr = self.RunGsUtil(['-o', 'GSUtil:stet_binary_path={}'.format(self.stet_binary_path), '-o', 'GSUtil:stet_config_path={}'.format(self.stet_config_path), 'cp', '--stet', suri(object_uri), test_file], return_stderr=True)
        self.assertNotIn('/4.0 B]', stderr)
        with open(test_file) as file_reader:
            downloaded_text = file_reader.read()
        self.assertIn('subcommand: decrypt', downloaded_text)
        self.assertIn('config file: --config-file={}'.format(self.stet_config_path), downloaded_text)
        self.assertIn('blob id: --blob-id={}'.format(suri(object_uri)), downloaded_text)
        self.assertIn('in file: {}'.format(test_file), downloaded_text)
        self.assertIn('out file: {}_.stet_tmp'.format(test_file), downloaded_text)
        self.assertFalse(os.path.exists(temporary_file_util.GetStetTempFileName(storage_url.StorageUrlFromString(test_file))))

    def test_does_not_seek_ahead_for_bytes_if_stet_transform(self):
        """Tests that cp does not seek-ahead for bytes if file size will change."""
        tmpdir = self.CreateTempDir()
        for _ in range(3):
            self.CreateTempFile(tmpdir=tmpdir, contents=b'123456')
        bucket_uri = self.CreateBucket()
        with util.SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', '-o', 'GSUtil:stet_binary_path={}'.format(self.stet_binary_path), '-o', 'GSUtil:stet_config_path={}'.format(self.stet_config_path), 'cp', '-r', '--stet', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.assertNotIn('18.0 B]', stderr)
            self.assertRegex(stderr, '2\\.\\d KiB]')