from __future__ import absolute_import
import os
import textwrap
from gslib.commands.rpo import RpoCommand
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
class TestRpoUnit(testcase.GsUtilUnitTestCase):

    def test_get_for_multiple_bucket_calls_api(self):
        bucket_uri1 = self.CreateBucket(bucket_name='rpofoo')
        bucket_uri2 = self.CreateBucket(bucket_name='rpobar')
        stdout = self.RunCommand('rpo', ['get', suri(bucket_uri1), suri(bucket_uri2)], return_stdout=True)
        expected_string = textwrap.dedent('      gs://rpofoo: None\n      gs://rpobar: None\n      ')
        self.assertEqual(expected_string, stdout)

    def test_get_with_wildcard(self):
        self.CreateBucket(bucket_name='boo1')
        self.CreateBucket(bucket_name='boo2')
        stdout = self.RunCommand('rpo', ['get', 'gs://boo*'], return_stdout=True)
        actual = '\n'.join(sorted(stdout.strip().split('\n')))
        expected_string = textwrap.dedent('      gs://boo1: None\n      gs://boo2: None')
        self.assertEqual(actual, expected_string)

    def test_get_with_wrong_url_raises_error(self):
        with self.assertRaisesRegex(CommandException, 'No URLs matched'):
            self.RunCommand('rpo', ['get', 'gs://invalid*'])

    def test_set_called_with_incorrect_value_raises_error(self):
        with self.assertRaisesRegex(CommandException, 'Invalid value for rpo set. Should be one of \\(ASYNC_TURBO\\|DEFAULT\\)'):
            self.RunCommand('rpo', ['set', 'random', 'gs://boo*'])

    def test_set_called_with_lower_case_value_raises_error(self):
        with self.assertRaisesRegex(CommandException, 'Invalid value for rpo set. Should be one of \\(ASYNC_TURBO\\|DEFAULT\\)'):
            self.RunCommand('rpo', ['set', 'async_turbo', 'gs://boo*'])

    def test_invalid_subcommand_raises_error(self):
        with self.assertRaisesRegex(CommandException, 'Invalid subcommand "blah", use get|set instead'):
            self.RunCommand('rpo', ['blah', 'DEFAULT', 'gs://boo*'])

    def test_shim_translates_recovery_point_objective_get_command(self):
        fake_cloudsdk_dir = 'fake_dir'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                self.CreateBucket(bucket_name='fake-bucket-get-rpo-1')
                mock_log_handler = self.RunCommand('rpo', args=['get', 'gs://fake-bucket-get-rpo-1'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets list --format=value[separator=": "](format("gs://{}", name),rpo.yesno(no="None")) --raw'.format(shim_util._get_gcloud_binary_path('fake_dir'), '{}'), info_lines)

    def test_shim_translates_recovery_point_objective_set_command(self):
        fake_cloudsdk_dir = 'fake_dir'
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': fake_cloudsdk_dir}):
                self.CreateBucket(bucket_name='fake-bucket-set-rpo')
                mock_log_handler = self.RunCommand('rpo', args=['set', 'DEFAULT', 'gs://fake-bucket-set-rpo'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --recovery-point-objective DEFAULT'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)