from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from gslib.commands import acl
from gslib.command import CreateOrGetGsutilLogger
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils import acl_helper
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import AclTranslation
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
class TestAclShim(testcase.GsUtilUnitTestCase):

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_get_object(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['get', 'gs://bucket/object'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects describe --format=multi(acl:format=json) --raw gs://bucket/object'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_get_bucket(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['get', 'gs://bucket'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets describe --format=multi(acl:format=json) --raw gs://bucket'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_set_object(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', inpath, 'gs://bucket/object'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --acl-file={}'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_set_bucket(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', inpath, 'gs://bucket'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --acl-file={} gs://bucket'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_predefined_acl_set_object(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', 'private', 'gs://bucket/object'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --predefined-acl=private gs://bucket/object'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_predefined_acl_set_bucket(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', 'private', 'gs://bucket'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --predefined-acl=private gs://bucket'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_xml_predefined_acl_for_set(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', 'public-read', 'gs://bucket'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --predefined-acl=publicRead gs://bucket'.format(shim_util._get_gcloud_binary_path('fake_dir')), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_set_multiple_buckets_urls(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', '-f', inpath, 'gs://bucket', 'gs://bucket1', 'gs://bucket2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --acl-file={} --continue-on-error gs://bucket gs://bucket1 gs://bucket2'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_set_multiple_objects_urls(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', '-f', inpath, 'gs://bucket/object', 'gs://bucket/object1', 'gs://bucket/object2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --acl-file={} --continue-on-error gs://bucket/object gs://bucket/object1 gs://bucket/object2'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_set_multiple_buckets_urls_recursive_all_versions(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['set', '-r', '-a', inpath, 'gs://bucket', 'gs://bucket1/o', 'gs://bucket2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --acl-file={} --recursive --all-versions gs://bucket gs://bucket1/o gs://bucket2'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_translates_acl_set_mix_buckets_and_objects_raises_error(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                with self.assertRaisesRegex(CommandException, 'Cannot operate on a mix of buckets and objects.'):
                    self.RunCommand('acl', ['set', 'acl-file', 'gs://bucket', 'gs://bucket1/object'])

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_bucket_acls_for_user(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['ch', '-u', 'user@example.com:R', 'gs://bucket1', 'gs://bucket2'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage buckets update --add-acl-grant entity=user-user@example.com,role=READER gs://bucket1 gs://bucket2'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_object_acls_for_user(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['ch', '-u', 'user@example.com:R', 'gs://bucket1/o', 'gs://bucket2/o'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --add-acl-grant entity=user-user@example.com,role=READER gs://bucket1/o gs://bucket2/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_raises_error_for_mix_of_objects_and_buckets(self):
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                with self.assertRaisesRegex(CommandException, 'Cannot operate on a mix of buckets and objects.'):
                    self.RunCommand('acl', ['ch', 'gs://bucket', 'gs://bucket1/object'])

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_acls_for_group(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['ch', '-g', 'group@example.com:W', 'gs://bucket1/o'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --add-acl-grant entity=group-group@example.com,role=WRITER gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_acls_for_domain(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['ch', '-g', 'example.com:O', 'gs://bucket1/o'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --add-acl-grant entity=domain-example.com,role=OWNER gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_acls_for_project(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['ch', '-p', 'owners-example:O', 'gs://bucket1/o'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --add-acl-grant entity=project-owners-example,role=OWNER gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_acls_for_all_users(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                for identifier in ['all', 'allUsers', 'AllUsers']:
                    mock_log_handler = self.RunCommand('acl', ['ch', '-g', identifier + ':O', 'gs://bucket1/o'], return_log_handler=True)
                    info_lines = '\n'.join(mock_log_handler.messages['info'])
                    self.assertIn('Gcloud Storage Command: {} storage objects update --add-acl-grant entity=allUsers,role=OWNER gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_changes_acls_for_all_authenticated_users(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                for identifier in ['allauth', 'allAuthenticatedUsers', 'AllAuthenticatedUsers']:
                    mock_log_handler = self.RunCommand('acl', ['ch', '-g', identifier + ':O', 'gs://bucket1/o'], return_log_handler=True)
                    info_lines = '\n'.join(mock_log_handler.messages['info'])
                    self.assertIn('Gcloud Storage Command: {} storage objects update --add-acl-grant entity=allAuthenticatedUsers,role=OWNER gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_deletes_acls(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('acl', ['ch', '-d', 'user@example.com', '-d', 'user1@example.com', 'gs://bucket1/o'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('Gcloud Storage Command: {} storage objects update --remove-acl-grant user@example.com --remove-acl-grant user1@example.com gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_removes_acls_for_all_users(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                for identifier in ['all', 'allUsers', 'AllUsers']:
                    mock_log_handler = self.RunCommand('acl', ['ch', '-d', identifier, 'gs://bucket1/o'], return_log_handler=True)
                    info_lines = '\n'.join(mock_log_handler.messages['info'])
                    self.assertIn('Gcloud Storage Command: {} storage objects update --remove-acl-grant AllUsers gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)

    @mock.patch.object(acl.AclCommand, 'RunCommand', new=mock.Mock())
    def test_shim_removes_acls_for_all_authenticated_users(self):
        inpath = self.CreateTempFile()
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                for identifier in ['allauth', 'allAuthenticatedUsers', 'AllAuthenticatedUsers']:
                    mock_log_handler = self.RunCommand('acl', ['ch', '-d', identifier, 'gs://bucket1/o'], return_log_handler=True)
                    info_lines = '\n'.join(mock_log_handler.messages['info'])
                    self.assertIn('Gcloud Storage Command: {} storage objects update --remove-acl-grant AllAuthenticatedUsers gs://bucket1/o'.format(shim_util._get_gcloud_binary_path('fake_dir'), inpath), info_lines)