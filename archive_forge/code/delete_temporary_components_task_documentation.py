from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
Deletes temporary components and associated tracker files.

    Args:
      task_status_queue: See base class.

    Returns:
      A task.Output with tasks for deleting temporary components.
    