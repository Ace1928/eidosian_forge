from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import upload_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _run_download(self, start_byte):
    """Performs the download operation."""
    request_config = request_config_factory.get_request_config(self._source_resource.storage_url, user_request_args=self._user_request_args)
    client = api_factory.get_api(self._source_resource.storage_url.scheme)
    try:
        if self._source_resource.size != 0:
            client.download_object(self._source_resource, self.writable_stream, request_config, start_byte=start_byte, download_strategy=cloud_api.DownloadStrategy.ONE_SHOT)
    except _AbruptShutdownError:
        pass
    except Exception as e:
        self.shutdown(e)