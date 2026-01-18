from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
def _MakeTask(self, source, dest):
    """Make a file copy Task for a single source.

    Args:
      source: paths.Path, The source file to copy.
      dest: path.Path, The destination to copy the file to.

    Raises:
      InvalidDestinationError: If this would end up copying to a path that has
        '.' or '..' as a segment.
      LocationMismatchError: If trying to copy a local file to a local file.

    Returns:
      storage_parallel.Task, The copy task to execute.
    """
    if not dest.IsPathSafe():
        raise InvalidDestinationError(source, dest)
    if source.is_remote:
        source_obj = storage_util.ObjectReference.FromUrl(source.path)
        if dest.is_remote:
            dest_obj = storage_util.ObjectReference.FromUrl(dest.path)
            return storage_parallel.FileRemoteCopyTask(source_obj, dest_obj)
        return storage_parallel.FileDownloadTask(source_obj, dest.path)
    if dest.is_remote:
        dest_obj = storage_util.ObjectReference.FromUrl(dest.path)
        return storage_parallel.FileUploadTask(source.path, dest_obj)
    raise LocationMismatchError('Cannot copy local file [{}] to local file [{}]'.format(source.path, dest.path))