from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import random
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import file_part_upload_task
from googlecloudsdk.command_lib.storage.tasks.cp import finalize_composite_upload_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Gzip the file at source_path necessary.

    Args:
      source_path (str): The source of the upload.
      temporary_paths_to_clean_up (list[str]): Adds the paths of any temporary
        files created to this list.

    Returns:
      The path to the gzipped temporary file if one was created. Otherwise,
        returns source_path.
    