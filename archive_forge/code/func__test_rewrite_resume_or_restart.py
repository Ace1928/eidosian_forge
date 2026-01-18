from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import unittest
from boto.storage_uri import BucketStorageUri
from gslib.cs_api_map import ApiSelector
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.gcs_json_api import GcsJsonApi
from gslib.project_id import PopulateProjectId
from gslib.tests.rewrite_helper import EnsureRewriteRestartCallbackHandler
from gslib.tests.rewrite_helper import EnsureRewriteResumeCallbackHandler
from gslib.tests.rewrite_helper import HaltingRewriteCallbackHandler
from gslib.tests.rewrite_helper import RewriteHaltException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import TEST_ENCRYPTION_KEY4
from gslib.tests.util import unittest
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.unit_util import ONE_MIB
def _test_rewrite_resume_or_restart(self, initial_dec_key, initial_enc_key, new_dec_key=None, new_enc_key=None):
    """Tests that the rewrite command restarts if the object's key changed.

    Args:
      initial_dec_key: Initial key the object is encrypted with, used as
          decryption key in the first rewrite call.
      initial_enc_key: Initial encryption key to rewrite the object with,
          used as encryption key in the first rewrite call.
      new_dec_key: Decryption key for the second rewrite call; if specified,
          object will be overwritten with a new encryption key in between
          the first and second rewrite calls, and this key will be used for
          the second rewrite call.
      new_enc_key: Encryption key for the second rewrite call; if specified,
          this key will be used for the second rewrite call, otherwise the
          initial key will be used.

    Returns:
      None
    """
    if self.test_api == ApiSelector.XML:
        return unittest.skip('Rewrite API is only supported in JSON.')
    bucket_uri = self.CreateBucket()
    destination_bucket_uri = self.CreateBucket(storage_class='NEARLINE')
    object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'12' * ONE_MIB + b'bar', prefer_json_api=True, encryption_key=initial_dec_key)
    destination_object_uri = self.CreateObject(bucket_uri=destination_bucket_uri, object_name='foo', contents='test', prefer_json_api=True, encryption_key=initial_dec_key)
    gsutil_api = GcsJsonApi(BucketStorageUri, logging.getLogger(), DiscardMessagesQueue(), self.default_provider)
    with SetBotoConfigForTest([('GSUtil', 'decryption_key1', initial_dec_key)]):
        src_obj_metadata = gsutil_api.GetObjectMetadata(object_uri.bucket_name, object_uri.object_name, provider=self.default_provider, fields=['bucket', 'contentType', 'etag', 'name'])
    dst_obj_metadata = gsutil_api.GetObjectMetadata(destination_object_uri.bucket_name, destination_object_uri.object_name, provider=self.default_provider, fields=['bucket', 'contentType', 'etag', 'name'])
    tracker_file_name = GetRewriteTrackerFilePath(src_obj_metadata.bucket, src_obj_metadata.name, dst_obj_metadata.bucket, dst_obj_metadata.name, self.test_api)
    decryption_tuple = CryptoKeyWrapperFromKey(initial_dec_key)
    decryption_tuple2 = CryptoKeyWrapperFromKey(new_dec_key or initial_dec_key)
    encryption_tuple = CryptoKeyWrapperFromKey(initial_enc_key)
    encryption_tuple2 = CryptoKeyWrapperFromKey(new_enc_key or initial_enc_key)
    try:
        try:
            gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata, progress_callback=HaltingRewriteCallbackHandler(ONE_MIB * 2).call, max_bytes_per_call=ONE_MIB, decryption_tuple=decryption_tuple, encryption_tuple=encryption_tuple)
            self.fail('Expected RewriteHaltException.')
        except RewriteHaltException:
            pass
        self.assertTrue(os.path.exists(tracker_file_name))
        if new_dec_key:
            self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'12' * ONE_MIB + b'bar', prefer_json_api=True, encryption_key=new_dec_key, gs_idempotent_generation=urigen(object_uri))
        with SetBotoConfigForTest([('GSUtil', 'decryption_key1', new_dec_key or initial_dec_key)]):
            original_md5 = gsutil_api.GetObjectMetadata(src_obj_metadata.bucket, src_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash
        if new_dec_key or new_enc_key:
            progress_callback = EnsureRewriteRestartCallbackHandler(ONE_MIB).call
        else:
            progress_callback = EnsureRewriteResumeCallbackHandler(ONE_MIB * 2).call
        gsutil_api.CopyObject(src_obj_metadata, dst_obj_metadata, progress_callback=progress_callback, max_bytes_per_call=ONE_MIB, decryption_tuple=decryption_tuple2, encryption_tuple=encryption_tuple2)
        self.assertFalse(os.path.exists(tracker_file_name))
        final_enc_key = new_enc_key or initial_enc_key
        with SetBotoConfigForTest([('GSUtil', 'encryption_key', final_enc_key)]):
            self.assertEqual(original_md5, gsutil_api.GetObjectMetadata(dst_obj_metadata.bucket, dst_obj_metadata.name, fields=['customerEncryption', 'md5Hash']).md5Hash, "Error: Rewritten object's hash doesn't match source object.")
    finally:
        DeleteTrackerFile(tracker_file_name)