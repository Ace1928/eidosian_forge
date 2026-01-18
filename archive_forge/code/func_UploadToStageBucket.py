from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import random
import re
import string
import time
from typing import Dict, Optional
from apitools.base.py import exceptions as http_exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
from apitools.base.py import util as http_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.functions import exceptions
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import http_client
from six.moves import range
def UploadToStageBucket(source_zip: str, function_ref: resources.Resource, stage_bucket: str) -> storage_util.ObjectReference:
    """Uploads the given source ZIP file to the provided staging bucket.

  Args:
    source_zip: the source ZIP file to upload.
    function_ref: the function resource reference.
    stage_bucket: the name of GCS bucket to stage the files to.

  Returns:
    dest_object: a reference to the uploaded Cloud Storage object.
  """
    zip_file = _GenerateRemoteZipFileName(function_ref)
    bucket_ref = storage_util.BucketReference.FromArgument(stage_bucket)
    dest_object = storage_util.ObjectReference.FromBucketRef(bucket_ref, zip_file)
    try:
        storage_api.StorageClient().CopyFileToGCS(source_zip, dest_object)
    except calliope_exceptions.BadFileException:
        raise exceptions.SourceUploadError('Failed to upload the function source code to the bucket {0}'.format(stage_bucket))
    return dest_object