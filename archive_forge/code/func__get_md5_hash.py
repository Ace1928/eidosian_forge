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
def _get_md5_hash(self):
    """Returns the MD5 Hash if present and hash validation is requested."""
    if properties.VALUES.storage.check_hashes.Get() == properties.CheckHashes.NEVER.value:
        return None
    if self._source_resource.md5_hash is None:
        log.warning('Found no hashes to validate object downloaded from %s and uploaded to %s. Integrity cannot be assured without hashes.', self._source_resource, self._destination_resource)
    return self._source_resource.md5_hash