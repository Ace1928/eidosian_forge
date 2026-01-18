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
def _send_manifest_message(task_status_queue, source_resource, destination_resource, result_status, md5_hash=None, description=None):
    """Send ManifestMessage to task_status_queue for processing."""
    task_status_queue.put(thread_messages.ManifestMessage(source_url=source_resource.storage_url, destination_url=destination_resource.storage_url, end_time=datetime.datetime.utcnow(), size=source_resource.size, result_status=result_status, md5_hash=md5_hash, description=description))