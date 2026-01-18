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
def _ExportGsutil(gcs_bucket, source, destination):
    """Exports files and directories from an environment's GCS bucket."""
    source_ref = storage_util.ObjectReference.FromBucketRef(gcs_bucket, source)
    if destination.startswith('gs://'):
        destination = _JoinPaths(destination.strip(posixpath.sep), '', gsutil_path=True)
    elif not os.path.isdir(destination):
        raise command_util.Error('Destination for export must be a directory.')
    try:
        retval = storage_util.RunGsutilCommand('cp', command_args=['-r', source_ref.ToUrl(), destination], run_concurrent=True, out_func=log.out.write, err_func=log.err.write)
    except (execution_utils.PermissionError, execution_utils.InvalidCommandError) as e:
        raise command_util.GsutilError(six.text_type(e))
    if retval:
        raise command_util.GsutilError('gsutil returned non-zero status code.')