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
def _gapfill_request_config_field(self, resource_args, request_config_field_name, source_resource_field_name):
    request_config_value = getattr(resource_args, request_config_field_name, None)
    if request_config_value is None:
        setattr(resource_args, request_config_field_name, getattr(self._source_resource, source_resource_field_name))