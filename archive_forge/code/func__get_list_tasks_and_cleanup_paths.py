from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import cp_command_util
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import rsync_command_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks import get_sorted_list_file_task
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
def _get_list_tasks_and_cleanup_paths(args, source_container, destination_container):
    """Generates tasks for creating source/destination inventories.

  Args:
    args (parser_extensions.Namespace): Command line arguments.
    source_container (resource_reference.Resource): Location to find source
      resources.
    destination_container (resource_reference.Resource): Location for
      destination resources.

  Returns:
    A tuple (list_tasks, cleanup_paths).
      list_tasks (List[task.Task]): The tasks to run to create inventories.
      cleanup_paths (List[str]): The paths where inventories are stored. The
        caller is responsible for removing these after transfers complete.
  """
    list_task_arguments = [(source_container, False), (destination_container, False)]
    if args.include_managed_folders:
        list_task_arguments.extend([(source_container, True), (destination_container, True)])
    cleanup_paths = []
    list_tasks = []
    for container, managed_folders_only in list_task_arguments:
        path = rsync_command_util.get_hashed_list_file_path(container.storage_url.url_string, is_managed_folder_list=managed_folders_only)
        cleanup_paths.append(path)
        task = get_sorted_list_file_task.GetSortedContainerContentsTask(container, path, exclude_pattern_strings=args.exclude, ignore_symlinks=args.ignore_symlinks, managed_folders_only=managed_folders_only, recurse=args.recursive)
        list_tasks.append(task)
    return (list_tasks, cleanup_paths)