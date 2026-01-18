from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import zip
def UploadDirectoryIfNecessary(path, staging_bucket=None, gs_prefix=None):
    """Uploads path to Cloud Storage if it isn't already there.

  Translates local file system paths to Cloud Storage-style paths (i.e. using
  the Unix path separator '/').

  Args:
    path: str, the path to the directory. Can be a Cloud Storage ("gs://") path
      or a local filesystem path (no protocol).
    staging_bucket: storage_util.BucketReference or None. If the path is local,
      the bucket to which it should be uploaded.
    gs_prefix: str, prefix for the directory within the staging bucket.

  Returns:
    str, a Cloud Storage path where the directory has been uploaded (possibly
      prior to the execution of this function).

  Raises:
    MissingStagingBucketException: if `path` is a local path, but staging_bucket
      isn't found.
    BadDirectoryError: if the given directory couldn't be found or is empty.
  """
    if path.startswith('gs://'):
        return path
    if staging_bucket is None:
        raise MissingStagingBucketException('The path provided was local, but no staging bucket for upload was provided.')
    if not os.path.isdir(path):
        raise BadDirectoryError('[{}] is not a valid directory.'.format(path))
    files = _GetFilesRelative(path)
    dests = [f.replace(_PATH_SEP, '/') for f in files]
    full_files = [_PATH_SEP.join([path, f]) for f in files]
    uploaded_paths = UploadFiles(list(zip(full_files, dests)), staging_bucket, gs_prefix=gs_prefix)
    if not uploaded_paths:
        raise BadDirectoryError('Cannot upload contents of directory [{}] to Google Cloud Storage; directory has no files.'.format(path))
    return uploaded_paths[0][:-len(dests[0])]