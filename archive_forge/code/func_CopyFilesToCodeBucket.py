from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import hashlib
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import times
from googlecloudsdk.third_party.appengine.tools import context_util
from six.moves import map  # pylint: disable=redefined-builtin
def CopyFilesToCodeBucket(upload_dir, source_files, bucket_ref, max_file_size=None):
    """Copies application files to the Google Cloud Storage code bucket.

  Use the Cloud Storage API using threads.

  Consider the following original structure:
    app/
      main.py
      tools/
        foo.py

   Assume main.py has SHA1 hash 123 and foo.py has SHA1 hash 456. The resultant
   GCS bucket will look like this:
     gs://$BUCKET/
       123
       456

   The resulting App Engine API manifest will be:
     {
       "app/main.py": {
         "sourceUrl": "https://storage.googleapis.com/staging-bucket/123",
         "sha1Sum": "123"
       },
       "app/tools/foo.py": {
         "sourceUrl": "https://storage.googleapis.com/staging-bucket/456",
         "sha1Sum": "456"
       }
     }

    A 'list' call of the bucket is made at the start, and files that hash to
    values already present in the bucket will not be uploaded again.

  Args:
    upload_dir: str, path to the service's upload directory
    source_files: [str], relative paths to upload.
    bucket_ref: The reference to the bucket files will be placed in.
    max_file_size: int, File size limit per individual file or None if no limit.

  Returns:
    A dictionary representing the manifest.
  """
    metrics.CustomTimedEvent(metric_names.COPY_APP_FILES_START)
    with file_utils.TemporaryDirectory() as tmp_dir:
        manifest = _BuildDeploymentManifest(upload_dir, source_files, bucket_ref, tmp_dir)
        files_to_upload = _BuildFileUploadMap(manifest, upload_dir, bucket_ref, tmp_dir, max_file_size)
        _UploadFilesThreads(files_to_upload, bucket_ref)
    log.status.Print('File upload done.')
    log.info('Manifest: [{0}]'.format(manifest))
    metrics.CustomTimedEvent(metric_names.COPY_APP_FILES)
    return manifest