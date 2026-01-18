from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import json
from apitools.base.py import encoding_helper
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
import six
def _call_appropriate_apitools_upload_strategy(self):
    """Calls StreamMedia, or StreamInChunks when the final size is unknown."""
    if self._should_gzip_in_flight:
        return self._apitools_upload.StreamInChunks()
    else:
        return self._apitools_upload.StreamMedia()