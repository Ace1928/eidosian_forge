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
def _get_upload(self):
    """Creates a new transfer object, or gets one from serialization data."""
    max_retries = properties.VALUES.storage.max_retries.GetInt()
    if self._serialization_data is not None:
        return transfer.Upload.FromData(self._source_stream, json.dumps(self._serialization_data), self._gcs_api.client.http, auto_transfer=False, gzip_encoded=self._should_gzip_in_flight, num_retries=max_retries)
    else:
        return super(__class__, self)._get_upload()