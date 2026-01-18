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
def _ImportStorageApi(gcs_bucket, source, destination):
    """Imports files and directories into a bucket."""
    client = storage_api.StorageClient()
    old_source = source
    source = source.rstrip('*')
    object_is_subdir = old_source != source
    if not object_is_subdir:
        source = source.rstrip(posixpath.sep)
    source_is_local = not source.startswith('gs://')
    if source_is_local and (not os.path.exists(source)):
        raise command_util.Error('Source for import does not exist.')
    source_dirname = _JoinPaths(os.path.dirname(source), '', gsutil_path=not source_is_local)
    if source_is_local:
        if os.path.isdir(source):
            file_chooser = gcloudignore.GetFileChooserForDir(source)
            for rel_path in file_chooser.GetIncludedFiles(source):
                file_path = _JoinPaths(source, rel_path)
                if os.path.isdir(file_path):
                    continue
                dest_path = _GetDestPath(source_dirname, file_path, destination, False)
                obj_ref = storage_util.ObjectReference.FromBucketRef(gcs_bucket, dest_path)
                client.CopyFileToGCS(file_path, obj_ref)
        else:
            dest_path = _GetDestPath(source_dirname, source, destination, False)
            obj_ref = storage_util.ObjectReference.FromBucketRef(gcs_bucket, dest_path)
            client.CopyFileToGCS(source, obj_ref)
    else:
        source_ref = storage_util.ObjectReference.FromUrl(source)
        to_import = _GetObjectOrSubdirObjects(source_ref, object_is_subdir=object_is_subdir, client=client)
        for obj in to_import:
            dest_object = storage_util.ObjectReference.FromBucketRef(gcs_bucket, _GetDestPath(source_dirname, obj.ToUrl(), destination, False))
            client.Copy(obj, dest_object)