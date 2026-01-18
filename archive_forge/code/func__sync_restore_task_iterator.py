from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.objects import bulk_restore_objects_task
from googlecloudsdk.command_lib.storage.tasks.objects import restore_object_task
from googlecloudsdk.core import log
def _sync_restore_task_iterator(args, fields_scope, user_request_args):
    """Yields blocking restore tasks."""
    last_resource = None
    for url in _url_iterator(args):
        resources = list(wildcard_iterator.get_wildcard_iterator(url.url_string, fields_scope=fields_scope, object_state=cloud_api.ObjectState.SOFT_DELETED))
        if not resources:
            raise errors.InvalidUrlError('The following URLs matched no objects:\n-{}'.format(url.url_string))
        for resource in resources:
            if args.all_versions:
                yield restore_object_task.RestoreObjectTask(resource, user_request_args)
            else:
                if last_resource and last_resource.storage_url.versionless_url_string != resource.storage_url.versionless_url_string:
                    yield restore_object_task.RestoreObjectTask(last_resource, user_request_args)
                last_resource = resource
    if last_resource:
        yield restore_object_task.RestoreObjectTask(last_resource, user_request_args)