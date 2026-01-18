from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
Make a file copy Task for a single source.

    Args:
      source: paths.Path, The source file to copy.
      dest: path.Path, The destination to copy the file to.

    Raises:
      InvalidDestinationError: If this would end up copying to a path that has
        '.' or '..' as a segment.
      LocationMismatchError: If trying to copy a local file to a local file.

    Returns:
      storage_parallel.Task, The copy task to execute.
    