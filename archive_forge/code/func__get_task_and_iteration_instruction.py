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
def _get_task_and_iteration_instruction(user_request_args, source_resource, source_container, destination_resource, destination_container, compare_only_hashes=False, delete_unmatched_destination_objects=False, dry_run=False, ignore_symlinks=False, skip_if_destination_has_later_modification_time=False, skip_unsupported=False):
    """Compares resources and returns next rsync step.

  Args:
    user_request_args (UserRequestArgs): User flags.
    source_resource: Source resource for comparison, a FileObjectResource,
      ManagedFolderResource, ObjectResource, or None. `None` indicates no
      sources left to copy.
    source_container (FileDirectoryResource|PrefixResource|BucketResource):
      Stripped from beginning of source_resource to get comparison URL.
    destination_resource: Destination resource for comparison, a
      FileObjectResource, ManagedFolderResource, ObjectResource, or None. `None`
      indicates all remaining source resources are new.
    destination_container (FileDirectoryResource|PrefixResource|BucketResource):
      If a copy task is generated for a source item with no equivalent existing
      destination item, it will copy to this general container. Also used to get
      comparison URL.
    compare_only_hashes (bool): Skip modification time comparison.
    delete_unmatched_destination_objects (bool): Clear objects at the
      destination that are not present at the source.
    dry_run (bool): Print what operations rsync would perform without actually
      executing them.
    ignore_symlinks (bool): Skip operations involving symlinks.
    skip_if_destination_has_later_modification_time (bool): Don't act if mtime
      metadata indicates we'd be overwriting with an older version of an object.
    skip_unsupported (bool): Skip copying unsupported object types.

  Returns:
    A pair of with a task and iteration instruction.

    First entry:
    None: Don't do anything for these resources.
    DeleteTask: Remove an extra resource from the destination.
    FileDownloadTask|FileUploadTask|IntraCloudCopyTask|ManagedFolderCopyTask:
      Update the destination with a copy of the source object.
    PatchFilePosixTask: Update the file destination POSIX data with the source's
      POSIX data.
    PatchObjectTask: Update the cloud destination's POSIX data with the source's
      POSIX data.

    Second entry:
    _IterateResource: Enum value indicating what to compare next.

  Raises:
    errors.Error: Missing a resource (does not account for subfunction errors).
  """
    if not (source_resource or destination_resource):
        raise errors.Error('Comparison requires at least a source or a destination.')
    if not source_resource:
        if delete_unmatched_destination_objects and (not isinstance(destination_resource, resource_reference.ManagedFolderResource)):
            if dry_run:
                _print_would_remove(destination_resource)
            else:
                return (_get_delete_task(destination_resource, user_request_args), _IterateResource.DESTINATION)
        return (None, _IterateResource.DESTINATION)
    if ignore_symlinks and source_resource.is_symlink:
        _log_skipping_symlink(source_resource)
        return (None, _IterateResource.SOURCE)
    if not isinstance(source_resource, resource_reference.ManagedFolderResource):
        source_posix = posix_util.get_posix_attributes_from_resource(source_resource)
        if user_request_args.preserve_posix:
            posix_to_set = source_posix
        else:
            posix_to_set = posix_util.PosixAttributes(None, source_posix.mtime, None, None, None)
    else:
        posix_to_set = None
    if not destination_resource:
        return (_get_copy_task(user_request_args, source_resource, posix_to_set=posix_to_set, source_container=source_container, destination_container=destination_container, dry_run=dry_run, skip_unsupported=skip_unsupported), _IterateResource.SOURCE)
    if ignore_symlinks and destination_resource.is_symlink:
        _log_skipping_symlink(destination_resource)
        return (None, _IterateResource.DESTINATION)
    source_url = _get_comparison_url(source_resource, source_container)
    destination_url = _get_comparison_url(destination_resource, destination_container)
    if source_url < destination_url:
        return (_get_copy_task(user_request_args, source_resource, posix_to_set=posix_to_set, source_container=source_container, destination_container=destination_container, dry_run=dry_run, skip_unsupported=skip_unsupported), _IterateResource.SOURCE)
    if source_url > destination_url:
        if delete_unmatched_destination_objects and (not isinstance(destination_resource, resource_reference.ManagedFolderResource)):
            if dry_run:
                _print_would_remove(destination_resource)
            else:
                return (_get_delete_task(destination_resource, user_request_args), _IterateResource.DESTINATION)
        return (None, _IterateResource.DESTINATION)
    if user_request_args.no_clobber:
        return (None, _IterateResource.SOURCE)
    if isinstance(source_resource, resource_reference.ManagedFolderResource):
        return (_get_copy_task(user_request_args, source_resource, source_container=source_container, destination_resource=destination_resource, destination_container=destination_container, dry_run=dry_run, posix_to_set=None, skip_unsupported=skip_unsupported), _IterateResource.BOTH)
    return _compare_equal_object_urls_to_get_task_and_iteration_instruction(user_request_args, source_resource, destination_resource, posix_to_set, compare_only_hashes=compare_only_hashes, dry_run=dry_run, skip_if_destination_has_later_modification_time=skip_if_destination_has_later_modification_time, skip_unsupported=skip_unsupported)