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
def _perform_component_download(self, request_config, progress_callback, digesters):
    """Component download does not validate hash or delete tracker."""
    destination_url = self._destination_resource.storage_url
    end_byte = self._offset + self._length - 1
    if self._strategy == cloud_api.DownloadStrategy.RESUMABLE:
        _, found_tracker_file = tracker_file_util.read_or_create_download_tracker_file(self._source_resource, destination_url, slice_start_byte=self._offset, component_number=self._component_number)
        first_null_byte = _get_first_null_byte_index(destination_url, offset=self._offset, length=self._length)
        start_byte = first_null_byte if found_tracker_file else self._offset
        if start_byte > end_byte:
            log.status.Print('{} component {} already downloaded.'.format(self._source_resource, self._component_number))
            self._calculate_deferred_hashes(digesters)
            self._catch_up_digesters(digesters, start_byte=self._offset, end_byte=self._source_resource.size)
            return
        if found_tracker_file and start_byte != self._offset:
            self._catch_up_digesters(digesters, start_byte=self._offset, end_byte=start_byte)
            log.status.Print('Resuming download for {} component {}'.format(self._source_resource, self._component_number))
    else:
        start_byte = self._offset
    return self._perform_download(request_config, progress_callback, self._disable_in_flight_decompression(True), self._strategy, start_byte, end_byte, files.BinaryFileWriterMode.MODIFY, digesters)