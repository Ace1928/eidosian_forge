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
def _compare_equal_object_urls_to_get_task_and_iteration_instruction(user_request_args, source_object, destination_object, posix_to_set, compare_only_hashes=False, dry_run=False, skip_if_destination_has_later_modification_time=False, skip_unsupported=False):
    """Similar to get_task_and_iteration_instruction except for equal URLs."""
    destination_posix = posix_util.get_posix_attributes_from_resource(destination_object)
    if skip_if_destination_has_later_modification_time and posix_to_set.mtime is not None and (destination_posix.mtime is not None) and (posix_to_set.mtime < destination_posix.mtime):
        return (None, _IterateResource.SOURCE)
    is_cloud_source_and_destination = isinstance(source_object, resource_reference.ObjectResource) and isinstance(destination_object, resource_reference.ObjectResource)
    if _compare_metadata_and_return_copy_needed(source_object, destination_object, posix_to_set.mtime, destination_posix.mtime, compare_only_hashes=compare_only_hashes, is_cloud_source_and_destination=is_cloud_source_and_destination):
        return (_get_copy_task(user_request_args, source_object, posix_to_set, destination_posix, destination_resource=destination_object, dry_run=dry_run, skip_unsupported=skip_unsupported), _IterateResource.BOTH)
    need_full_posix_update = user_request_args.preserve_posix and posix_to_set != destination_posix
    need_mtime_update = not is_cloud_source_and_destination and posix_to_set.mtime is not None and (posix_to_set.mtime != destination_posix.mtime)
    if not (need_full_posix_update or need_mtime_update):
        return (None, _IterateResource.BOTH)
    if dry_run:
        if need_full_posix_update:
            log.status.Print('Would set POSIX attributes for {}'.format(destination_object))
        else:
            log.status.Print('Would set mtime for {}'.format(destination_object))
        return (None, _IterateResource.BOTH)
    if isinstance(destination_object, resource_reference.ObjectResource):
        return (patch_object_task.PatchObjectTask(destination_object, posix_to_set=posix_to_set, user_request_args=user_request_args), _IterateResource.BOTH)
    return (patch_file_posix_task.PatchFilePosixTask(posix_util.get_system_posix_data(), source_object, destination_object, posix_to_set, destination_posix), _IterateResource.BOTH)