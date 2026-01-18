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
@SkipForS3('Tests use GS ACL model.')
class TestAcl(TestAclBase):
    """Integration tests for acl command."""

    def setUp(self):
        super(TestAcl, self).setUp()
        self.sample_uri = self.CreateBucket()
        self.sample_url = StorageUrlFromString(str(self.sample_uri))
        self.logger = CreateOrGetGsutilLogger('acl')
        self._project_number = self.json_api.GetBucket(self.CreateBucket().bucket_name, fields=['projectNumber']).projectNumber
        self._project_test_acl = '%s-%s' % (self._project_team, self._project_number)

    def test_set_invalid_acl_object(self):
        """Ensures that invalid content returns a bad request error."""
        obj_uri = suri(self.CreateObject(contents=b'foo'))
        inpath = self.CreateTempFile(contents=b'badAcl')
        stderr = self.RunGsUtil(self._set_acl_prefix + [inpath, obj_uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            error_text = 'Found invalid JSON/YAML file'
        else:
            error_text = 'ArgumentException'
        self.assertIn(error_text, stderr)

    def test_set_invalid_acl_bucket(self):
        """Ensures that invalid content returns a bad request error."""
        bucket_uri = suri(self.CreateBucket())
        inpath = self.CreateTempFile(contents=b'badAcl')
        stderr = self.RunGsUtil(self._set_acl_prefix + [inpath, bucket_uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            error_text = 'Found invalid JSON/YAML file'
        else:
            error_text = 'ArgumentException'
        self.assertIn(error_text, stderr)

    def test_set_xml_acl_json_api_object(self):
        """Ensures XML content returns a bad request error and migration warning."""
        obj_uri = suri(self.CreateObject(contents=b'foo'))
        inpath = self.CreateTempFile(contents=b'<ValidXml></ValidXml>')
        stderr = self.RunGsUtil(self._set_acl_prefix + [inpath, obj_uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('Found invalid JSON/YAML file', stderr)
        else:
            self.assertIn('ArgumentException', stderr)
            self.assertIn('XML ACL data provided', stderr)

    def test_set_xml_acl_json_api_bucket(self):
        """Ensures XML content returns a bad request error and migration warning."""
        bucket_uri = suri(self.CreateBucket())
        inpath = self.CreateTempFile(contents=b'<ValidXml></ValidXml>')
        stderr = self.RunGsUtil(self._set_acl_prefix + [inpath, bucket_uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('Found invalid JSON/YAML file', stderr)
        else:
            self.assertIn('ArgumentException', stderr)
            self.assertIn('XML ACL data provided', stderr)

    def test_set_valid_acl_object(self):
        """Tests setting a valid ACL on an object."""
        obj_uri = suri(self.CreateObject(contents=b'foo'))
        acl_string = self.RunGsUtil(self._get_acl_prefix + [obj_uri], return_stdout=True)
        inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
        self.RunGsUtil(self._set_acl_prefix + ['public-read', obj_uri])
        acl_string2 = self.RunGsUtil(self._get_acl_prefix + [obj_uri], return_stdout=True)
        self.RunGsUtil(self._set_acl_prefix + [inpath, obj_uri])
        acl_string3 = self.RunGsUtil(self._get_acl_prefix + [obj_uri], return_stdout=True)
        self.assertNotEqual(acl_string, acl_string2)
        self.assertEqual(acl_string, acl_string3)

    def test_set_valid_permission_whitespace_object(self):
        """Ensures that whitespace is allowed in role and entity elements."""
        obj_uri = suri(self.CreateObject(contents=b'foo'))
        acl_string = self.RunGsUtil(self._get_acl_prefix + [obj_uri], return_stdout=True)
        acl_string = re.sub('"role"', '"role" \\n', acl_string)
        acl_string = re.sub('"entity"', '\\n "entity"', acl_string)
        inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
        self.RunGsUtil(self._set_acl_prefix + [inpath, obj_uri])

    def test_set_valid_acl_bucket(self):
        """Ensures that valid canned and XML ACLs work with get/set."""
        if self._ServiceAccountCredentialsPresent():
            return unittest.skip('Canned ACLs orphan service account permissions.')
        bucket_uri = suri(self.CreateBucket())
        acl_string = self.RunGsUtil(self._get_acl_prefix + [bucket_uri], return_stdout=True)
        inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
        self.RunGsUtil(self._set_acl_prefix + ['public-read', bucket_uri])
        acl_string2 = self.RunGsUtil(self._get_acl_prefix + [bucket_uri], return_stdout=True)
        self.RunGsUtil(self._set_acl_prefix + [inpath, bucket_uri])
        acl_string3 = self.RunGsUtil(self._get_acl_prefix + [bucket_uri], return_stdout=True)
        self.assertNotEqual(acl_string, acl_string2)
        self.assertEqual(acl_string, acl_string3)

    def test_invalid_canned_acl_object(self):
        """Ensures that an invalid canned ACL returns a CommandException."""
        obj_uri = suri(self.CreateObject(contents=b'foo'))
        stderr = self.RunGsUtil(self._set_acl_prefix + ['not-a-canned-acl', obj_uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('AttributeError', stderr)
        else:
            self.assertIn('CommandException', stderr)
            self.assertIn('Invalid canned ACL', stderr)

    def test_set_valid_def_acl_bucket(self):
        """Ensures that valid default canned and XML ACLs works with get/set."""
        bucket_uri = self.CreateBucket()
        obj_uri1 = suri(self.CreateObject(bucket_uri=bucket_uri, contents=b'foo'))
        acl_string = self.RunGsUtil(self._get_acl_prefix + [obj_uri1], return_stdout=True)
        self.RunGsUtil(self._set_defacl_prefix + ['authenticated-read', suri(bucket_uri)])

        @Retry(AssertionError, tries=5, timeout_secs=1)
        def _Check1():
            obj_uri2 = suri(self.CreateObject(bucket_uri=bucket_uri, contents=b'foo2'))
            acl_string2 = self.RunGsUtil(self._get_acl_prefix + [obj_uri2], return_stdout=True)
            self.assertNotEqual(acl_string, acl_string2)
            self.assertIn('allAuthenticatedUsers', acl_string2)
        _Check1()
        inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
        self.RunGsUtil(self._set_defacl_prefix + [inpath, suri(bucket_uri)])

        @Retry(AssertionError, tries=5, timeout_secs=1)
        def _Check2():
            obj_uri3 = suri(self.CreateObject(bucket_uri=bucket_uri, contents=b'foo3'))
            acl_string3 = self.RunGsUtil(self._get_acl_prefix + [obj_uri3], return_stdout=True)
            self.assertEqual(acl_string, acl_string3)
        _Check2()

    def test_acl_set_version_specific_uri(self):
        """Tests setting an ACL on a specific version of an object."""
        bucket_uri = self.CreateVersionedBucket()
        uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data')
        inpath = self.CreateTempFile(contents=b'def')
        self.RunGsUtil(['cp', inpath, uri.uri])
        lines = self.AssertNObjectsInBucket(bucket_uri, 2, versioned=True)
        v0_uri_str, v1_uri_str = (lines[0], lines[1])
        orig_acls = []
        for uri_str in (v0_uri_str, v1_uri_str):
            acl = self.RunGsUtil(self._get_acl_prefix + [uri_str], return_stdout=True)
            self.assertNotIn(PUBLIC_READ_JSON_ACL_TEXT, self._strip_json_whitespace(acl))
            orig_acls.append(acl)
        self.RunGsUtil(self._set_acl_prefix + ['public-read', v0_uri_str])
        acl = self.RunGsUtil(self._get_acl_prefix + [v0_uri_str], return_stdout=True)
        self.assertIn(PUBLIC_READ_JSON_ACL_TEXT, self._strip_json_whitespace(acl))
        acl = self.RunGsUtil(self._get_acl_prefix + [v1_uri_str], return_stdout=True)
        self.assertNotIn(PUBLIC_READ_JSON_ACL_TEXT, self._strip_json_whitespace(acl))
        acl = self.RunGsUtil(self._get_acl_prefix + [uri.uri], return_stdout=True)
        self.assertEqual(acl, orig_acls[0])

    def _strip_json_whitespace(self, json_text):
        return re.sub('\\s*', '', json_text)

    def _MakeScopeRegex(self, role, entity_type, email_address):
        template_regex = '\\{.*"entity":\\s*"%s-%s".*"role":\\s*"%s".*\\}' % (entity_type, email_address, role)
        return re.compile(template_regex, flags=re.DOTALL)

    def _MakeProjectScopeRegex(self, role, project_team, project_number):
        template_regex = '\\{.*"entity":\\s*"project-%s-%s",\\s*"projectTeam":\\s*\\{\\s*"projectNumber":\\s*"%s",\\s*"team":\\s*"%s"\\s*\\},\\s*"role":\\s*"%s".*\\}' % (project_team, project_number, project_number, project_team, role)
        return re.compile(template_regex, flags=re.DOTALL)

    def testAclChangeWithUserId(self):
        test_regex = self._MakeScopeRegex('READER', 'user', self.USER_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-u', self.USER_TEST_ID + ':r', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', self.USER_TEST_ADDRESS, suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testAclChangeWithGroupId(self):
        test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', self.GROUP_TEST_ID + ':r', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', self.GROUP_TEST_ADDRESS, suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testAclChangeWithUserEmail(self):
        test_regex = self._MakeScopeRegex('READER', 'user', self.USER_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-u', self.USER_TEST_ADDRESS + ':r', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', self.USER_TEST_ADDRESS, suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testAclChangeWithGroupEmail(self):
        test_regex = self._MakeScopeRegex('OWNER', 'group', self.GROUP_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':fc', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', self.GROUP_TEST_ADDRESS, suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testAclChangeWithDomain(self):
        test_regex = self._MakeScopeRegex('READER', 'domain', self.DOMAIN_TEST)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', self.DOMAIN_TEST + ':r', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', self.DOMAIN_TEST, suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    @SkipForXML('XML API does not support project scopes.')
    def testAclChangeWithProjectOwners(self):
        test_regex = self._MakeProjectScopeRegex('WRITER', self._project_team, self._project_number)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-p', self._project_test_acl + ':w', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)

    def testAclChangeWithAllUsers(self):
        test_regex = re.compile('\\{.*"entity":\\s*"allUsers".*"role":\\s*"WRITER".*\\}', flags=re.DOTALL)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', 'allusers' + ':w', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', 'allusers', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testAclChangeWithAllAuthUsers(self):
        test_regex = re.compile('\\{.*"entity":\\s*"allAuthenticatedUsers".*"role":\\s*"READER".*\\}', flags=re.DOTALL)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', 'allauthenticatedusers' + ':r', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', 'allauthenticatedusers', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testBucketAclChange(self):
        """Tests acl change on a bucket."""
        test_regex = self._MakeScopeRegex('OWNER', 'user', self.USER_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-u', self.USER_TEST_ADDRESS + ':fc', suri(self.sample_uri)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        test_regex2 = self._MakeScopeRegex('WRITER', 'user', self.USER_TEST_ADDRESS)
        s1, s2 = self.RunGsUtil(self._ch_acl_prefix + ['-u', self.USER_TEST_ADDRESS + ':w', suri(self.sample_uri)], return_stderr=True, return_stdout=True)
        json_text2 = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertRegex(json_text2, test_regex2)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', self.USER_TEST_ADDRESS, suri(self.sample_uri)])
        json_text3 = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
        self.assertNotRegex(json_text3, test_regex)

    def testProjectAclChangesOnBucket(self):
        """Tests project entity acl changes on a bucket."""
        if self.test_api == ApiSelector.XML:
            stderr = self.RunGsUtil(self._ch_acl_prefix + ['-p', self._project_test_acl + ':w', suri(self.sample_uri)], expected_status=1, return_stderr=True)
            self.assertIn('CommandException: XML API does not support project scopes, cannot translate ACL.', stderr)
        else:
            test_regex = self._MakeProjectScopeRegex('WRITER', self._project_team, self._project_number)
            self.RunGsUtil(self._ch_acl_prefix + ['-p', self._project_test_acl + ':w', suri(self.sample_uri)])
            json_text = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
            self.assertRegex(json_text, test_regex)
            self.RunGsUtil(self._ch_acl_prefix + ['-d', self._project_test_acl, suri(self.sample_uri)])
            json_text2 = self.RunGsUtil(self._get_acl_prefix + [suri(self.sample_uri)], return_stdout=True)
            self.assertNotRegex(json_text2, test_regex)

    def testObjectAclChange(self):
        """Tests acl change on an object."""
        obj = self.CreateObject(bucket_uri=self.sample_uri, contents=b'something')
        self.AssertNObjectsInBucket(self.sample_uri, 1)
        test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', suri(obj)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        test_regex2 = self._MakeScopeRegex('OWNER', 'group', self.GROUP_TEST_ADDRESS)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':OWNER', suri(obj)])
        json_text2 = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertRegex(json_text2, test_regex2)
        self.RunGsUtil(self._ch_acl_prefix + ['-d', self.GROUP_TEST_ADDRESS, suri(obj)])
        json_text3 = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertNotRegex(json_text3, test_regex2)
        all_auth_regex = re.compile('\\{.*"entity":\\s*"allAuthenticatedUsers".*"role":\\s*"OWNER".*\\}', flags=re.DOTALL)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', 'AllAuth:O', suri(obj)])
        json_text4 = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertRegex(json_text4, all_auth_regex)

    def testObjectAclChangeAllUsers(self):
        """Tests acl ch AllUsers:R on an object."""
        obj = self.CreateObject(bucket_uri=self.sample_uri, contents=b'something')
        self.AssertNObjectsInBucket(self.sample_uri, 1)
        all_users_regex = re.compile('\\{.*"entity":\\s*"allUsers".*"role":\\s*"READER".*\\}', flags=re.DOTALL)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertNotRegex(json_text, all_users_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', 'AllUsers:R', suri(obj)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertRegex(json_text, all_users_regex)

    def testSeekAheadAcl(self):
        """Tests seek-ahead iterator with ACL sub-commands."""
        object_uri = self.CreateObject(contents=b'foo')
        current_acl = self.RunGsUtil(['acl', 'get', suri(object_uri)], return_stdout=True)
        current_acl_file = self.CreateTempFile(contents=current_acl.encode(UTF8))
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'acl', 'ch', '-u', 'AllUsers:R', suri(object_uri)], return_stderr=True)
            self.assertIn('Estimated work for this command: objects: 1\n', stderr)
            stderr = self.RunGsUtil(['-m', 'acl', 'set', current_acl_file, suri(object_uri)], return_stderr=True)
            self.assertIn('Estimated work for this command: objects: 1\n', stderr)
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '0'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'acl', 'ch', '-u', 'AllUsers:R', suri(object_uri)], return_stderr=True)
            self.assertNotIn('Estimated work', stderr)

    def testMultithreadedAclChange(self, count=10):
        """Tests multi-threaded acl changing on several objects."""
        objects = []
        for i in range(count):
            objects.append(self.CreateObject(bucket_uri=self.sample_uri, contents='something {0}'.format(i).encode('ascii')))
        self.AssertNObjectsInBucket(self.sample_uri, count)
        test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        json_texts = []
        for obj in objects:
            json_texts.append(self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True))
        for json_text in json_texts:
            self.assertNotRegex(json_text, test_regex)
        uris = [suri(obj) for obj in objects]
        self.RunGsUtil(['-m', '-DD'] + self._ch_acl_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ'] + uris)
        json_texts = []
        for obj in objects:
            json_texts.append(self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True))
        for json_text in json_texts:
            self.assertRegex(json_text, test_regex)

    def testRecursiveChangeAcl(self):
        """Tests recursively changing ACLs on nested objects."""
        obj = self.CreateObject(bucket_uri=self.sample_uri, object_name='foo/bar', contents=b'something')
        self.AssertNObjectsInBucket(self.sample_uri, 1)
        test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

        @Retry(AssertionError, tries=5, timeout_secs=1)
        def _AddAcl():
            self.RunGsUtil(self._ch_acl_prefix + ['-R', '-g', self.GROUP_TEST_ADDRESS + ':READ', suri(obj)[:-3]])
            json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
            self.assertRegex(json_text, test_regex)
        _AddAcl()

        @Retry(AssertionError, tries=5, timeout_secs=1)
        def _DeleteAcl():
            delete_grant = self.GROUP_TEST_ADDRESS.upper()
            self.RunGsUtil(self._ch_acl_prefix + ['-d', delete_grant, suri(obj)])
            json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
            self.assertNotRegex(json_text, test_regex)
        _DeleteAcl()

    def testMultiVersionSupport(self):
        """Tests changing ACLs on multiple object versions."""
        bucket = self.CreateVersionedBucket()
        object_name = self.MakeTempName('obj')
        obj1_uri = self.CreateObject(bucket_uri=bucket, object_name=object_name, contents=b'One thing')
        self.CreateObject(bucket_uri=bucket, object_name=object_name, contents=b'Another thing', gs_idempotent_generation=urigen(obj1_uri))
        lines = self.AssertNObjectsInBucket(bucket, 2, versioned=True)
        obj_v1, obj_v2 = (lines[0], lines[1])
        test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._get_acl_prefix + [obj_v1], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._ch_acl_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', obj_v1])
        json_text = self.RunGsUtil(self._get_acl_prefix + [obj_v1], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        json_text = self.RunGsUtil(self._get_acl_prefix + [obj_v2], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testBadRequestAclChange(self):
        stdout, stderr = self.RunGsUtil(self._ch_acl_prefix + ['-u', 'invalid_$$@hello.com:R', suri(self.sample_uri)], return_stdout=True, return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('HTTPError', stderr)
        else:
            self.assertIn('BadRequestException', stderr)
        self.assertNotIn('Retrying', stdout)
        self.assertNotIn('Retrying', stderr)

    def testAclGetWithoutFullControl(self):
        object_uri = self.CreateObject(contents=b'foo')
        expected_error_regex = 'Anonymous \\S+ do(es)? not have'
        with self.SetAnonymousBotoCreds():
            stderr = self.RunGsUtil(self._get_acl_prefix + [suri(object_uri)], return_stderr=True, expected_status=1)
            self.assertRegex(stderr, expected_error_regex)

    def testTooFewArgumentsFails(self):
        """Tests calling ACL commands with insufficient number of arguments."""
        stderr = self.RunGsUtil(self._get_acl_prefix, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(self._set_acl_prefix, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(self._ch_acl_prefix, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(['acl'], return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)

    def testMinusF(self):
        """Tests -f option to continue after failure."""
        bucket_uri = self.CreateBucket()
        obj_uri = suri(self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'foo'))
        acl_string = self.RunGsUtil(self._get_acl_prefix + [obj_uri], return_stdout=True)
        self.RunGsUtil(['-d'] + self._set_acl_prefix + ['-f', 'public-read', obj_uri + 'foo2', obj_uri], expected_status=1)
        acl_string2 = self.RunGsUtil(self._get_acl_prefix + [obj_uri], return_stdout=True)
        self.assertNotEqual(acl_string, acl_string2)