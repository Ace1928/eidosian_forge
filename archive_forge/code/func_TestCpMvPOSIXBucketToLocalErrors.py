from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import ast
import base64
import binascii
import datetime
import gzip
import logging
import os
import pickle
import pkgutil
import random
import re
import stat
import string
import sys
import threading
from unittest import mock
from apitools.base.py import exceptions as apitools_exceptions
import boto
from boto import storage_uri
from boto.exception import ResumableTransferDisposition
from boto.exception import StorageResponseError
from boto.storage_uri import BucketStorageUri
from gslib import command
from gslib import exception
from gslib import name_expansion
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.cs_api_map import ApiSelector
from gslib.daisy_chain_wrapper import _DEFAULT_DOWNLOAD_CHUNK_SIZE
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import InvalidUrlError
from gslib.gcs_json_api import GcsJsonApi
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
from gslib.tests.rewrite_helper import EnsureRewriteResumeCallbackHandler
from gslib.tests.rewrite_helper import HaltingRewriteCallbackHandler
from gslib.tests.rewrite_helper import RewriteHaltException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import NotParallelizable
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import HaltingCopyCallbackHandler
from gslib.tests.util import HaltOneComponentCopyCallbackHandler
from gslib.tests.util import HAS_GS_PORT
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import KmsTestingResources
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import ORPHANED_FILE
from gslib.tests.util import POSIX_GID_ERROR
from gslib.tests.util import POSIX_INSUFFICIENT_ACCESS_ERROR
from gslib.tests.util import POSIX_MODE_ERROR
from gslib.tests.util import POSIX_UID_ERROR
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TailSet
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import GetSlicedDownloadTrackerFilePaths
from gslib.ui_controller import BytesToFixedWidthString
from gslib.utils import hashing_helper
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.constants import UTF8
from gslib.utils.copy_helper import GetTrackerFilePath
from gslib.utils.copy_helper import PARALLEL_UPLOAD_STATIC_SALT
from gslib.utils.copy_helper import PARALLEL_UPLOAD_TEMP_NAMESPACE
from gslib.utils.copy_helper import TrackerFileType
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.posix_util import ValidatePOSIXMode
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.text_util import get_random_ascii_chars
from gslib.utils.unit_util import EIGHT_MIB
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from gslib.utils import shim_util
import six
from six.moves import http_client
from six.moves import range
from six.moves import xrange
def TestCpMvPOSIXBucketToLocalErrors(cls, bucket_uri, obj, tmpdir, is_cp=True):
    """Helper function for preserve_posix_errors tests in test_cp and test_mv.

  Args:
    cls: An instance of either TestCp or TestMv.
    bucket_uri: The uri of the bucket that the object is in.
    obj: The object to run the tests on.
    tmpdir: The local file path to cp to.
    is_cp: Whether or not the calling test suite is cp or mv.
  """
    error_key = 'error_regex'
    if cls._use_gcloud_storage:
        insufficient_access_error = no_read_access_error = re.compile('User \\d+ owns file, but owner does not have read permission')
        missing_gid_error = re.compile("GID in .* metadata doesn't exist on current system")
        missing_uid_error = re.compile("UID in .* metadata doesn't exist on current system")
    else:
        insufficient_access_error = BuildErrorRegex(obj, POSIX_INSUFFICIENT_ACCESS_ERROR)
        missing_gid_error = BuildErrorRegex(obj, POSIX_GID_ERROR)
        missing_uid_error = BuildErrorRegex(obj, POSIX_UID_ERROR)
        no_read_access_error = BuildErrorRegex(obj, POSIX_MODE_ERROR)
    test_params = {'test1': {MODE_ATTR: '333', error_key: no_read_access_error}, 'test2': {GID_ATTR: GetInvalidGid, error_key: missing_gid_error}, 'test3': {GID_ATTR: GetInvalidGid, MODE_ATTR: '420', error_key: missing_gid_error}, 'test4': {UID_ATTR: INVALID_UID, error_key: missing_uid_error}, 'test5': {UID_ATTR: INVALID_UID, MODE_ATTR: '530', error_key: missing_uid_error}, 'test6': {UID_ATTR: INVALID_UID, GID_ATTR: GetInvalidGid, error_key: missing_uid_error}, 'test7': {UID_ATTR: INVALID_UID, GID_ATTR: GetInvalidGid, MODE_ATTR: '640', error_key: missing_uid_error}, 'test8': {UID_ATTR: INVALID_UID, GID_ATTR: GetPrimaryGid, error_key: missing_uid_error}, 'test9': {UID_ATTR: INVALID_UID, GID_ATTR: GetNonPrimaryGid, error_key: missing_uid_error}, 'test10': {UID_ATTR: INVALID_UID, GID_ATTR: GetPrimaryGid, MODE_ATTR: '640', error_key: missing_uid_error}, 'test11': {UID_ATTR: INVALID_UID, GID_ATTR: GetNonPrimaryGid, MODE_ATTR: '640', error_key: missing_uid_error}, 'test12': {UID_ATTR: USER_ID, GID_ATTR: GetInvalidGid, error_key: missing_gid_error}, 'test13': {UID_ATTR: USER_ID, GID_ATTR: GetInvalidGid, MODE_ATTR: '640', error_key: missing_gid_error}, 'test14': {GID_ATTR: GetPrimaryGid, MODE_ATTR: '240', error_key: insufficient_access_error}}
    for test_name, attrs_dict in six.iteritems(test_params):
        cls.ClearPOSIXMetadata(obj)
        uid = attrs_dict.get(UID_ATTR)
        if uid is not None and callable(uid):
            uid = uid()
        gid = attrs_dict.get(GID_ATTR)
        if gid is not None and callable(gid):
            gid = gid()
        mode = attrs_dict.get(MODE_ATTR)
        cls.SetPOSIXMetadata(cls.default_provider, bucket_uri.bucket_name, obj.object_name, uid=uid, gid=gid, mode=mode)
        stderr = cls.RunGsUtil(['cp' if is_cp else 'mv', '-P', suri(bucket_uri, obj.object_name), tmpdir], expected_status=1, return_stderr=True)
        if cls._use_gcloud_storage:
            general_posix_error = 'ERROR'
        else:
            general_posix_error = ORPHANED_FILE
        cls.assertIn(general_posix_error, stderr, 'Error during test "%s": %s not found in stderr:\n%s' % (test_name, general_posix_error, stderr))
        error_regex = attrs_dict[error_key]
        cls.assertTrue(error_regex.search(stderr), 'Test %s did not match expected error; could not find a match for %s\n\nin stderr:\n%s' % (test_name, error_regex.pattern, stderr))
        listing1 = TailSet(suri(bucket_uri), cls.FlatListBucket(bucket_uri))
        listing2 = TailSet(tmpdir, cls.FlatListDir(tmpdir))
        cls.assertEqual(listing1, set(['/%s' % obj.object_name]))
        cls.assertEqual(listing2, set(['']))