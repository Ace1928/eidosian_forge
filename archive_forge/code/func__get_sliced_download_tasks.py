from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.command_lib.storage.tasks.cp import file_part_download_task
from googlecloudsdk.command_lib.storage.tasks.cp import finalize_sliced_download_task
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
def _get_sliced_download_tasks(self):
    """Creates all tasks necessary for a sliced download."""
    component_offsets_and_lengths = copy_component_util.get_component_offsets_and_lengths(self._source_resource.size, copy_component_util.get_component_count(self._source_resource.size, properties.VALUES.storage.sliced_object_download_component_size.Get(), properties.VALUES.storage.sliced_object_download_max_components.GetInt()))
    download_component_task_list = []
    for i, (offset, length) in enumerate(component_offsets_and_lengths):
        download_component_task_list.append(file_part_download_task.FilePartDownloadTask(self._source_resource, self._temporary_destination_resource, offset=offset, length=length, component_number=i, total_components=len(component_offsets_and_lengths), strategy=self._strategy, user_request_args=self._user_request_args))
    finalize_sliced_download_task_list = [finalize_sliced_download_task.FinalizeSlicedDownloadTask(self._source_resource, self._temporary_destination_resource, self._destination_resource, delete_source=self._delete_source, do_not_decompress=self._do_not_decompress, posix_to_set=self._posix_to_set, system_posix_data=self._system_posix_data, user_request_args=self._user_request_args)]
    return (download_component_task_list, finalize_sliced_download_task_list)