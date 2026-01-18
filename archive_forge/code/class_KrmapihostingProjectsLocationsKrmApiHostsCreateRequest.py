from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KrmapihostingProjectsLocationsKrmApiHostsCreateRequest(_messages.Message):
    """A KrmapihostingProjectsLocationsKrmApiHostsCreateRequest object.

  Fields:
    krmApiHost: A KrmApiHost resource to be passed as the request body.
    krmApiHostId: Required. Client chosen ID for the KrmApiHost.
    parent: Required. The parent in whose context the KrmApiHost is created.
      The parent value is in the format:
      'projects/{project_id}/locations/{location}'.
    requestId: Optional. A unique ID to identify requests. This is unique such
      that if the request is re-tried, the server will know to ignore the
      request if it has already been completed. The server will guarantee that
      for at least 60 minutes after the first request. The request ID must be
      a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    krmApiHost = _messages.MessageField('KrmApiHost', 1)
    krmApiHostId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)