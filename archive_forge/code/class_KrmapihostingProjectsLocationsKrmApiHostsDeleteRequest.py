from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KrmapihostingProjectsLocationsKrmApiHostsDeleteRequest(_messages.Message):
    """A KrmapihostingProjectsLocationsKrmApiHostsDeleteRequest object.

  Fields:
    name: Required. The name of this service resource in the format: 'projects
      /{project_id}/locations/{location}/krmApiHosts/{krm_api_host_id}'.
    requestId: Optional. A unique ID to identify requests. This is unique such
      that if the request is re-tried, the server will know to ignore the
      request if it has already been completed. The server will guarantee that
      for at least 60 minutes after the first request. The request ID must be
      a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)