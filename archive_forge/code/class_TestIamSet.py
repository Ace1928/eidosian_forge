from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('Tests use GS IAM model.')
@SkipForXML('XML IAM control is not supported.')
class TestIamSet(TestIamIntegration):
    """Integration tests for iam set command."""

    def setUp(self):
        super(TestIamSet, self).setUp()
        self.public_bucket_read_binding = gen_binding(IAM_BUCKET_READ_ROLE)
        self.public_object_read_binding = gen_binding(IAM_OBJECT_READ_ROLE)
        self.project_viewer_objectviewer_with_cond_binding = gen_binding(IAM_OBJECT_VIEWER_ROLE, members=['projectViewer:%s' % PopulateProjectId()], condition={'title': TEST_CONDITION_TITLE, 'description': TEST_CONDITION_DESCRIPTION, 'expression': TEST_CONDITION_EXPR_RESOURCE_IS_OBJECT})
        self.bucket = self.CreateBucket()
        self.versioned_bucket = self.CreateVersionedBucket()
        self.bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.old_bucket_iam_path = self.CreateTempFile(contents=self.bucket_iam_string.encode(UTF8))
        self.new_bucket_iam_policy = patch_binding(json.loads(self.bucket_iam_string), IAM_BUCKET_READ_ROLE, self.public_bucket_read_binding)
        self.new_bucket_iam_path = self.CreateTempFile(contents=json.dumps(self.new_bucket_iam_policy).encode(UTF8))
        self.new_bucket_policy_with_conditions_policy = json.loads(self.bucket_iam_string)
        self.new_bucket_policy_with_conditions_policy['bindings'].append(self.project_viewer_objectviewer_with_cond_binding[0])
        self.new_bucket_policy_with_conditions_path = self.CreateTempFile(contents=json.dumps(self.new_bucket_policy_with_conditions_policy))
        self.object = self.CreateObject(contents='foobar')
        self.object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
        self.old_object_iam_path = self.CreateTempFile(contents=self.object_iam_string.encode(UTF8))
        self.new_object_iam_policy = patch_binding(json.loads(self.object_iam_string), IAM_OBJECT_READ_ROLE, self.public_object_read_binding)
        self.new_object_iam_path = self.CreateTempFile(contents=json.dumps(self.new_object_iam_policy).encode(UTF8))

    def test_seek_ahead_iam(self):
        """Ensures that the seek-ahead iterator is being used with iam commands."""
        gsutil_object = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'iam', 'set', self.new_object_iam_path, gsutil_object.uri], return_stderr=True)
            self.assertIn('Estimated work for this command: objects: 1\n', stderr)

    def test_set_mix_of_buckets_and_objects(self):
        """Tests that failure is thrown when buckets and objects are provided."""
        stderr = self.RunGsUtil(['iam', 'set', self.new_object_iam_path, self.bucket.uri, self.object.uri], return_stderr=True, expected_status=1)
        self.assertIn('CommandException', stderr)

    def test_set_file_url(self):
        """Tests that failure is thrown when a file url is provided."""
        stderr = self.RunGsUtil(['iam', 'set', self.new_object_iam_path, 'file://somefile'], return_stderr=True, expected_status=1)
        self.assertIn('AttributeError', stderr)

    def test_set_invalid_iam_bucket(self):
        """Ensures invalid content returns error on input check."""

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            inpath = self.CreateTempFile(contents=b'badIam')
            stderr = self.RunGsUtil(['iam', 'set', inpath, self.bucket.uri], return_stderr=True, expected_status=1)
            error_message = 'Found invalid JSON/YAML' if self._use_gcloud_storage else 'ArgumentException'
            self.assertIn(error_message, stderr)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stderr = self.RunGsUtil(['iam', 'set', 'nonexistent/path', self.bucket.uri], return_stderr=True, expected_status=1)
            error_message = 'No such file or directory' if self._use_gcloud_storage else 'ArgumentException'
            self.assertIn(error_message, stderr)
        _Check1()
        _Check2()

    def test_get_invalid_bucket(self):
        """Ensures that invalid bucket names returns an error."""
        stderr = self.RunGsUtil(['iam', 'get', self.nonexistent_bucket_name], return_stderr=True, expected_status=1)
        error_message = 'AttributeError' if self._use_gcloud_storage else 'CommandException'
        self.assertIn(error_message, stderr)
        stderr = self.RunGsUtil(['iam', 'get', 'gs://%s' % self.nonexistent_bucket_name], return_stderr=True, expected_status=1)
        error_message = 'not found' if self._use_gcloud_storage else 'BucketNotFoundException'
        self.assertIn(error_message, stderr)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check():
            stderr = self.RunGsUtil(['iam', 'get', 'gs://*'], return_stderr=True, expected_status=1)
            error_message = 'must match a single cloud resource' if self._use_gcloud_storage else 'CommandException'
            self.assertIn(error_message, stderr)
        _Check()

    def test_set_valid_iam_bucket(self):
        """Tests setting a valid IAM on a bucket."""
        self.RunGsUtil(['iam', 'set', '-e', '', self.new_bucket_iam_path, self.bucket.uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.RunGsUtil(['iam', 'set', '-e', '', self.old_bucket_iam_path, self.bucket.uri])
        reset_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.bucket_iam_string, reset_iam_string)
        self.assertIn(self.public_bucket_read_binding[0], json.loads(set_iam_string)['bindings'])

    @unittest.skip('Disabled until all projects whitelisted for conditions.')
    def test_set_and_get_valid_bucket_policy_with_conditions(self):
        """Tests setting and getting an IAM policy with conditions on a bucket."""
        self.RunGsUtil(['iam', 'set', '-e', '', self.new_bucket_policy_with_conditions_path, self.bucket.uri])
        get_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertIn(TEST_CONDITION_DESCRIPTION, get_iam_string)
        self.assertIn(TEST_CONDITION_EXPR_RESOURCE_IS_OBJECT, get_iam_string.replace('\\', ''))
        self.assertIn(TEST_CONDITION_TITLE, get_iam_string)

    @unittest.skip('Disabled until all projects whitelisted for conditions.')
    def test_ch_fails_after_setting_conditions(self):
        """Tests that if we "set" a policy with conditions, "ch" won't patch it."""
        print()
        self.RunGsUtil(['iam', 'set', '-e', '', self.new_bucket_policy_with_conditions_path, self.bucket.uri])
        stderr = self.RunGsUtil(['iam', 'ch', 'allUsers:objectViewer', self.bucket.uri], return_stderr=True, expected_status=1)
        self.assertIn('CommandException: Could not patch IAM policy for', stderr)
        self.assertIn('The resource had conditions present', stderr)
        stderr = self.RunGsUtil(['iam', 'ch', '-f', 'allUsers:objectViewer', self.bucket.uri], return_stderr=True, expected_status=1)
        self.assertIn('CommandException: Some IAM policies could not be patched', stderr)
        self.assertIn('Some resources had conditions', stderr)

    def test_set_blank_etag(self):
        """Tests setting blank etag behaves appropriately."""
        self.RunGsUtil(['iam', 'set', '-e', '', self.new_bucket_iam_path, self.bucket.uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.RunGsUtil(['iam', 'set', '-e', json.loads(set_iam_string)['etag'], self.old_bucket_iam_path, self.bucket.uri])
        reset_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.bucket_iam_string, reset_iam_string)
        self.assertIn(self.public_bucket_read_binding[0], json.loads(set_iam_string)['bindings'])

    def test_set_valid_etag(self):
        """Tests setting valid etag behaves correctly."""
        get_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.RunGsUtil(['iam', 'set', '-e', json.loads(get_iam_string)['etag'], self.new_bucket_iam_path, self.bucket.uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.RunGsUtil(['iam', 'set', '-e', json.loads(set_iam_string)['etag'], self.old_bucket_iam_path, self.bucket.uri])
        reset_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.bucket_iam_string, reset_iam_string)
        self.assertIn(self.public_bucket_read_binding[0], json.loads(set_iam_string)['bindings'])

    def test_set_invalid_etag(self):
        """Tests setting an invalid etag format raises an error."""
        self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        stderr = self.RunGsUtil(['iam', 'set', '-e', 'some invalid etag', self.new_bucket_iam_path, self.bucket.uri], return_stderr=True, expected_status=1)
        error_message = 'DecodeError' if self._use_gcloud_storage else 'ArgumentException'
        self.assertIn(error_message, stderr)

    def test_set_mismatched_etag(self):
        """Tests setting mismatched etag raises an error."""
        get_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.RunGsUtil(['iam', 'set', '-e', json.loads(get_iam_string)['etag'], self.new_bucket_iam_path, self.bucket.uri])
        stderr = self.RunGsUtil(['iam', 'set', '-e', json.loads(get_iam_string)['etag'], self.new_bucket_iam_path, self.bucket.uri], return_stderr=True, expected_status=1)
        error_message = 'pre-conditions you specified did not hold' if self._use_gcloud_storage else 'PreconditionException'
        self.assertIn(error_message, stderr)

    def _create_multiple_objects(self):
        """Creates two versioned objects and return references to all versions.

    Returns:
      A four-tuple (a, b, a*, b*) of storage_uri.BucketStorageUri instances.
    """
        old_gsutil_object = self.CreateObject(bucket_uri=self.versioned_bucket, contents=b'foo')
        old_gsutil_object2 = self.CreateObject(bucket_uri=self.versioned_bucket, contents=b'bar')
        gsutil_object = self.CreateObject(bucket_uri=self.versioned_bucket, object_name=old_gsutil_object.object_name, contents=b'new_foo', gs_idempotent_generation=urigen(old_gsutil_object))
        gsutil_object2 = self.CreateObject(bucket_uri=self.versioned_bucket, object_name=old_gsutil_object2.object_name, contents=b'new_bar', gs_idempotent_generation=urigen(old_gsutil_object2))
        return (old_gsutil_object, old_gsutil_object2, gsutil_object, gsutil_object2)

    def test_set_valid_iam_multiple_objects(self):
        """Tests setting a valid IAM on multiple objects."""
        old_gsutil_object, old_gsutil_object2, gsutil_object, gsutil_object2 = self._create_multiple_objects()
        self.RunGsUtil(['iam', 'set', '-r', self.new_object_iam_path, self.versioned_bucket.uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', gsutil_object.uri], return_stdout=True)
        set_iam_string2 = self.RunGsUtil(['iam', 'get', gsutil_object2.uri], return_stdout=True)
        self.assertEqualsPoliciesString(set_iam_string, set_iam_string2)
        self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])
        iam_string_old = self.RunGsUtil(['iam', 'get', old_gsutil_object.version_specific_uri], return_stdout=True)
        iam_string_old2 = self.RunGsUtil(['iam', 'get', old_gsutil_object2.version_specific_uri], return_stdout=True)
        self.assertEqualsPoliciesString(iam_string_old, iam_string_old2)
        self.assertEqualsPoliciesString(self.object_iam_string, iam_string_old)

    def test_set_valid_iam_multithreaded_multiple_objects(self):
        """Tests setting a valid IAM on multiple objects."""
        old_gsutil_object, old_gsutil_object2, gsutil_object, gsutil_object2 = self._create_multiple_objects()
        self.RunGsUtil(['-m', 'iam', 'set', '-r', self.new_object_iam_path, self.versioned_bucket.uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', gsutil_object.uri], return_stdout=True)
        set_iam_string2 = self.RunGsUtil(['iam', 'get', gsutil_object2.uri], return_stdout=True)
        self.assertEqualsPoliciesString(set_iam_string, set_iam_string2)
        self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])
        iam_string_old = self.RunGsUtil(['iam', 'get', old_gsutil_object.version_specific_uri], return_stdout=True)
        iam_string_old2 = self.RunGsUtil(['iam', 'get', old_gsutil_object2.version_specific_uri], return_stdout=True)
        self.assertEqualsPoliciesString(iam_string_old, iam_string_old2)
        self.assertEqualsPoliciesString(self.object_iam_string, iam_string_old)

    def test_set_valid_iam_multiple_objects_all_versions(self):
        """Tests set IAM policy on all versions of all objects."""
        old_gsutil_object, old_gsutil_object2, gsutil_object, gsutil_object2 = self._create_multiple_objects()
        self.RunGsUtil(['iam', 'set', '-ra', self.new_object_iam_path, self.versioned_bucket.uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', gsutil_object.version_specific_uri], return_stdout=True)
        set_iam_string2 = self.RunGsUtil(['iam', 'get', gsutil_object2.version_specific_uri], return_stdout=True)
        set_iam_string_old = self.RunGsUtil(['iam', 'get', old_gsutil_object.version_specific_uri], return_stdout=True)
        set_iam_string_old2 = self.RunGsUtil(['iam', 'get', old_gsutil_object2.version_specific_uri], return_stdout=True)
        self.assertEqualsPoliciesString(set_iam_string, set_iam_string2)
        self.assertEqualsPoliciesString(set_iam_string, set_iam_string_old)
        self.assertEqualsPoliciesString(set_iam_string, set_iam_string_old2)
        self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])

    def test_set_error(self):
        """Tests fail-fast behavior of iam set.

    We initialize two buckets (bucket, bucket2) and attempt to set both along
    with a third, non-existent bucket in between, self.nonexistent_bucket_name.

    We want to ensure
      1.) Bucket "bucket" IAM policy has been set appropriately,
      2.) Bucket self.nonexistent_bucket_name has caused an error, and
      3.) gsutil has exited and "bucket2"'s IAM policy is unaltered.
    """
        bucket = self.CreateBucket()
        bucket2 = self.CreateBucket()
        stderr = self.RunGsUtil(['iam', 'set', '-e', '', self.new_bucket_iam_path, bucket.uri, 'gs://%s' % self.nonexistent_bucket_name, bucket2.uri], return_stderr=True, expected_status=1)
        error_message = 'not found' if self._use_gcloud_storage else 'BucketNotFoundException'
        self.assertIn(error_message, stderr)
        set_iam_string = self.RunGsUtil(['iam', 'get', bucket.uri], return_stdout=True)
        set_iam_string2 = self.RunGsUtil(['iam', 'get', bucket2.uri], return_stdout=True)
        self.assertIn(self.public_bucket_read_binding[0], json.loads(set_iam_string)['bindings'])
        self.assertEqualsPoliciesString(self.bucket_iam_string, set_iam_string2)

    def test_set_force_error(self):
        """Tests ignoring failure behavior of iam set.

    Similar to TestIamSet.test_set_error, except here we want to ensure
      1.) Bucket "bucket" IAM policy has been set appropriately,
      2.) Bucket self.nonexistent_bucket_name has caused an error, BUT
      3.) gsutil has continued and "bucket2"'s IAM policy has been set as well.
    """
        bucket = self.CreateBucket()
        bucket2 = self.CreateBucket()
        stderr = self.RunGsUtil(['iam', 'set', '-f', self.new_bucket_iam_path, bucket.uri, 'gs://%s' % self.nonexistent_bucket_name, bucket2.uri], return_stderr=True, expected_status=1)
        error_message = 'not found' if self._use_gcloud_storage else 'CommandException'
        self.assertIn(error_message, stderr)
        set_iam_string = self.RunGsUtil(['iam', 'get', bucket.uri], return_stdout=True)
        set_iam_string2 = self.RunGsUtil(['iam', 'get', bucket2.uri], return_stdout=True)
        self.assertIn(self.public_bucket_read_binding[0], json.loads(set_iam_string)['bindings'])
        self.assertEqualsPoliciesString(set_iam_string, set_iam_string2)

    def test_set_multithreaded_error(self):
        """Tests fail-fast behavior of multithreaded iam set.

    This is testing gsutil iam set with the -m and -r flags present in
    invocation.

    N.B.: Currently, (-m, -r) behaves identically to (-m, -fr) and (-fr,).
    However, (-m, -fr) and (-fr,) behavior is not as expected due to
    name_expansion.NameExpansionIterator.next raising problematic e.g. 404
    or 403 errors. More details on this issue can be found in comments in
    commands.iam.IamCommand._SetIam.

    Thus, the following command
      gsutil -m iam set -fr <object_policy> gs://bad_bucket gs://good_bucket

    will NOT set policies on objects in gs://good_bucket due to an error when
    iterating over gs://bad_bucket.
    """

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stderr = self.RunGsUtil(['-m', 'iam', 'set', '-r', self.new_object_iam_path, 'gs://%s' % self.nonexistent_bucket_name, self.bucket.uri], return_stderr=True, expected_status=1)
            error_message = 'not found' if self._use_gcloud_storage else 'BucketNotFoundException'
            self.assertIn(error_message, stderr)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            gsutil_object = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
            gsutil_object2 = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
            set_iam_string = self.RunGsUtil(['iam', 'get', gsutil_object.uri], return_stdout=True)
            set_iam_string2 = self.RunGsUtil(['iam', 'get', gsutil_object2.uri], return_stdout=True)
            self.assertEqualsPoliciesString(set_iam_string, set_iam_string2)
            self.assertEqualsPoliciesString(self.object_iam_string, set_iam_string)
        _Check1()
        _Check2()

    def test_set_valid_iam_single_unversioned_object(self):
        """Tests setting a valid IAM on an object."""
        gsutil_object = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
        lookup_uri = gsutil_object.uri
        self.RunGsUtil(['iam', 'set', self.new_object_iam_path, lookup_uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.RunGsUtil(['iam', 'set', '-e', json.loads(set_iam_string)['etag'], self.old_object_iam_path, lookup_uri])
        reset_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.object_iam_string, reset_iam_string)
        self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])

    def test_set_valid_iam_single_versioned_object(self):
        """Tests setting a valid IAM on a versioned object."""
        gsutil_object = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
        lookup_uri = gsutil_object.version_specific_uri
        self.RunGsUtil(['iam', 'set', self.new_object_iam_path, lookup_uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.RunGsUtil(['iam', 'set', '-e', json.loads(set_iam_string)['etag'], self.old_object_iam_path, lookup_uri])
        reset_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.object_iam_string, reset_iam_string)
        self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])

    def test_set_valid_iam_multithreaded_single_object(self):
        """Tests setting a valid IAM on a single object with multithreading."""
        gsutil_object = self.CreateObject(bucket_uri=self.bucket, contents=b'foobar')
        lookup_uri = gsutil_object.version_specific_uri
        self.RunGsUtil(['-m', 'iam', 'set', '-e', '', self.new_object_iam_path, lookup_uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.RunGsUtil(['-m', 'iam', 'set', '-e', '', self.old_object_iam_path, lookup_uri])
        reset_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.object_iam_string, reset_iam_string)
        self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])
        lookup_uri = '%s*' % self.bucket.uri
        self.RunGsUtil(['-m', 'iam', 'set', '-e', '', self.new_object_iam_path, lookup_uri])
        set_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.RunGsUtil(['-m', 'iam', 'set', '-e', '', self.old_object_iam_path, lookup_uri])
        reset_iam_string = self.RunGsUtil(['iam', 'get', lookup_uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.object_iam_string, reset_iam_string)
        self.assertIn(self.public_object_read_binding[0], json.loads(set_iam_string)['bindings'])