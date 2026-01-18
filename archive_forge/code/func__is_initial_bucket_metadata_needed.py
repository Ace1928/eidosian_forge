from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.buckets import update_bucket_task
def _is_initial_bucket_metadata_needed(user_request_args):
    """Determines if the bucket update has to patch existing metadata."""
    resource_args = user_request_args.resource_args
    if not resource_args:
        return False
    return user_request_args_factory.adds_or_removes_acls(user_request_args) or any([resource_args.labels_file_path, resource_args.labels_to_append, resource_args.labels_to_remove])