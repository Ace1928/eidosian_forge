from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.command_lib.storage.tasks.cp import file_part_task
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
def _perform_download(self, request_config, progress_callback, do_not_decompress, download_strategy, start_byte, end_byte, write_mode, digesters):
    """Prepares file stream, calls API, and validates hash."""
    with files.BinaryFileWriter(self._destination_resource.storage_url.object_name, create_path=True, mode=write_mode, convert_invalid_windows_characters=properties.VALUES.storage.convert_incompatible_windows_path_characters.GetBool()) as download_stream:
        download_stream.seek(start_byte)
        provider = self._source_resource.storage_url.scheme
        api_download_result = api_factory.get_api(provider).download_object(self._source_resource, download_stream, request_config, digesters=digesters, do_not_decompress=do_not_decompress, download_strategy=download_strategy, progress_callback=progress_callback, start_byte=start_byte, end_byte=end_byte)
    self._calculate_deferred_hashes(digesters)
    if hash_util.HashAlgorithm.MD5 in digesters:
        calculated_digest = hash_util.get_base64_hash_digest_string(digesters[hash_util.HashAlgorithm.MD5])
        download_util.validate_download_hash_and_delete_corrupt_files(self._destination_resource.storage_url.object_name, self._source_resource.md5_hash, calculated_digest)
    elif hash_util.HashAlgorithm.CRC32C in digesters:
        if self._component_number is None:
            calculated_digest = crc32c.get_hash(digesters[hash_util.HashAlgorithm.CRC32C])
            download_util.validate_download_hash_and_delete_corrupt_files(self._destination_resource.storage_url.object_name, self._source_resource.crc32c_hash, calculated_digest)
    return api_download_result