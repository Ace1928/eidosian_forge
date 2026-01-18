from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsEndpointAttachmentsDeleteRequest(_messages.Message):
    """A ConnectorsProjectsLocationsEndpointAttachmentsDeleteRequest object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/endpointAttachments/*`
  """
    name = _messages.StringField(1, required=True)