from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.dataproc import exceptions as dp_exceptions
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six.moves.urllib.parse
def _UploadStorageClient(files, destination, storage_client=None):
    """Upload a list of local files to GCS.

  Args:
    files: The list of local files to upload.
    destination: A GCS "directory" to copy the files into.
    storage_client: Storage api client used to copy files to gcs.
  """
    client = storage_client or storage_api.StorageClient()
    for file_to_upload in files:
        file_name = os.path.basename(file_to_upload)
        dest_url = os.path.join(destination, file_name)
        dest_object = storage_util.ObjectReference.FromUrl(dest_url)
        try:
            client.CopyFileToGCS(file_to_upload, dest_object)
        except exceptions.BadFileException as err:
            raise dp_exceptions.FileUploadError("Failed to upload files ['{}'] to '{}': {}".format("', '".join(files), destination, err))