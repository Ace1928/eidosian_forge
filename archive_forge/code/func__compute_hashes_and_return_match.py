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
def _compute_hashes_and_return_match(source_resource, destination_resource):
    """Does minimal computation to compare checksums of resources."""
    if source_resource.size != destination_resource.size:
        return False
    check_hashes = properties.VALUES.storage.check_hashes.Get()
    if check_hashes == properties.CheckHashes.NEVER.value:
        return True
    for resource in (source_resource, destination_resource):
        if isinstance(resource, resource_reference.ObjectResource) and resource.crc32c_hash is resource.md5_hash is None:
            log.warning('Found no hashes to validate on {}. Will not copy unless file modification time or size difference.'.format(resource.storage_url.versionless_url_string))
            return True
    if isinstance(source_resource, resource_reference.ObjectResource) and isinstance(destination_resource, resource_reference.ObjectResource):
        source_crc32c = source_resource.crc32c_hash
        destination_crc32c = destination_resource.crc32c_hash
        source_md5 = source_resource.md5_hash
        destination_md5 = destination_resource.md5_hash
        log.debug('Comparing hashes for two cloud objects. CRC32C checked first. If no comparable hash pairs, will not copy.\n{}:\n  CRC32C: {}\n  MD5: {}\n{}:\n  CRC32C: {}\n  MD5: {}\n'.format(source_resource.storage_url.versionless_url_string, source_crc32c, source_md5, destination_resource.storage_url.versionless_url_string, destination_crc32c, destination_md5))
        if source_crc32c is not None and destination_crc32c is not None:
            return source_crc32c == destination_crc32c
        if source_md5 is not None and destination_md5 is not None:
            return source_md5 == destination_md5
        return True
    is_upload = isinstance(source_resource, resource_reference.FileObjectResource)
    if is_upload:
        cloud_resource = destination_resource
        local_resource = source_resource
    else:
        cloud_resource = source_resource
        local_resource = destination_resource
    if cloud_resource.crc32c_hash is not None and cloud_resource.md5_hash is None:
        fast_crc32c_util.log_or_raise_crc32c_issues(warn_for_always=is_upload)
        if not fast_crc32c_util.check_if_will_use_fast_crc32c(install_if_missing=True) and check_hashes == properties.CheckHashes.IF_FAST_ELSE_SKIP.value:
            return True
        compare_crc32c = True
    elif cloud_resource.crc32c_hash is not None:
        compare_crc32c = fast_crc32c_util.check_if_will_use_fast_crc32c(install_if_missing=False)
    else:
        compare_crc32c = False
    if compare_crc32c:
        hash_algorithm = hash_util.HashAlgorithm.CRC32C
        cloud_hash = cloud_resource.crc32c_hash
    else:
        hash_algorithm = hash_util.HashAlgorithm.MD5
        cloud_hash = cloud_resource.md5_hash
    local_hash = hash_util.get_base64_hash_digest_string(hash_util.get_hash_from_file(local_resource.storage_url.object_name, hash_algorithm))
    return cloud_hash == local_hash