from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import csv
import datetime
import enum
import os
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
class ManifestManager:
    """Handles writing copy statuses to manifest."""

    def __init__(self, manifest_path):
        """Creates manifest file with correct headers."""
        self._manifest_column_headers = ['Source', 'Destination', 'Start', 'End', 'Md5'] + (['UploadId'] if properties.VALUES.storage.run_by_gsutil_shim.GetBool() else []) + ['Source Size', 'Bytes Transferred', 'Result', 'Description']
        self._manifest_path = manifest_path
        if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
            return
        with files.FileWriter(manifest_path, newline='\n') as file_writer:
            csv.DictWriter(file_writer, self._manifest_column_headers).writeheader()

    def write_row(self, manifest_message, file_progress=None):
        """Writes data to manifest file."""
        if file_progress and manifest_message.result_status is ResultStatus.OK:
            bytes_copied = file_progress.total_bytes_copied
        else:
            bytes_copied = 0
        end_time = manifest_message.end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        if file_progress:
            start_time = file_progress.start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            start_time = end_time
        if manifest_message.description:
            description = manifest_message.description.replace('\n', '\\n').replace('\r', '\\r')
        else:
            description = ''
        row_dictionary = {'Source': manifest_message.source_url.url_string, 'Destination': manifest_message.destination_url.versionless_url_string, 'Start': start_time, 'End': end_time, 'Md5': manifest_message.md5_hash or '', 'Source Size': manifest_message.size, 'Bytes Transferred': bytes_copied, 'Result': manifest_message.result_status.value, 'Description': description}
        if properties.VALUES.storage.run_by_gsutil_shim.GetBool():
            row_dictionary['UploadId'] = None
        with files.FileWriter(self._manifest_path, append=True, newline='\n') as file_writer:
            csv.DictWriter(file_writer, self._manifest_column_headers).writerow(row_dictionary)