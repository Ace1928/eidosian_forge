from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
def _GetFileCopyTasks(self, sources, dest):
    """Get the Tasks to be executed to copy the given sources.

    If dest is dir-like (ending in a slash), all sources are copied under the
    destination. If it is file-like, at most one source can be provided and it
    is copied directly to the destination name.

    Args:
      sources: [paths.Path], The source files to copy. These must all be files
        not directories.
      dest: paths.Path, The destination to copy the files to.

    Returns:
      [storage_parallel.Task], The file copy tasks to execute.
    """
    if not sources:
        return []
    tasks = []
    for source in sources:
        item_dest = self._GetDestinationName(source, dest)
        tasks.append(self._MakeTask(source, item_dest))
    return tasks