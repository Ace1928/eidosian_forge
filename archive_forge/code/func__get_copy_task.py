from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import path_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks import patch_file_posix_task
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.objects import patch_object_task
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _get_copy_task(user_request_args, source_resource, posix_to_set=None, source_container=None, destination_resource=None, destination_container=None, dry_run=False, skip_unsupported=False):
    """Generates copy tasks with generic settings and logic."""
    if skip_unsupported:
        unsupported_type = resource_util.get_unsupported_object_type(source_resource)
        if unsupported_type:
            log.status.Print(resource_util.UNSUPPORTED_OBJECT_WARNING_FORMAT.format(source_resource, unsupported_type.value))
            return
    if destination_resource:
        copy_destination = destination_resource
    else:
        copy_destination = _get_copy_destination_resource(source_resource, source_container, destination_container)
    if dry_run:
        if isinstance(source_resource, resource_reference.FileObjectResource):
            try:
                with files.BinaryFileReader(source_resource.storage_url.object_name):
                    pass
            except:
                log.error('Could not open {}'.format(source_resource.storage_url.object_name))
                raise
        log.status.Print('Would copy {} to {}'.format(source_resource, copy_destination))
        return
    if isinstance(source_resource, resource_reference.CloudResource) and (isinstance(destination_container, resource_reference.CloudResource) or isinstance(destination_resource, resource_reference.CloudResource)):
        if user_request_args.resource_args and user_request_args.resource_args.preserve_acl:
            fields_scope = cloud_api.FieldsScope.FULL
        else:
            fields_scope = cloud_api.FieldsScope.RSYNC
    else:
        fields_scope = None
    return copy_task_factory.get_copy_task(source_resource, copy_destination, do_not_decompress=True, fetch_source_fields_scope=fields_scope, posix_to_set=posix_to_set, user_request_args=user_request_args, verbose=True)