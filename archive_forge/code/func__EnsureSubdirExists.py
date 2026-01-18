from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os.path
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _EnsureSubdirExists(bucket_ref, subdir):
    """Checks that a directory marker object exists in the bucket or creates one.

  The directory marker object is needed for subdir listing to not crash
  if the directory is empty.

  Args:
    bucket_ref: googlecloudsk.api_lib.storage.storage_util.BucketReference,
        a reference to the environment's bucket
    subdir: str, the subdirectory to check or recreate. Should not contain
        slashes.
  """
    subdir_name = '{}/'.format(subdir)
    subdir_ref = storage_util.ObjectReference.FromBucketRef(bucket_ref, subdir_name)
    storage_client = storage_api.StorageClient()
    try:
        storage_client.GetObject(subdir_ref)
    except apitools_exceptions.HttpNotFoundError:
        insert_req = storage_client.messages.StorageObjectsInsertRequest(bucket=bucket_ref.bucket, name=subdir_name)
        upload = transfer.Upload.FromStream(io.BytesIO(), 'application/octet-stream')
        try:
            storage_client.client.objects.Insert(insert_req, upload=upload)
        except apitools_exceptions.HttpError:
            raise command_util.Error('Error re-creating empty {}/ directory most likely due to lack of permissions.'.format(subdir))
    except apitools_exceptions.HttpForbiddenError:
        raise command_util.Error('Error checking directory {}/ marker object exists due to lack of permissions.'.format(subdir))