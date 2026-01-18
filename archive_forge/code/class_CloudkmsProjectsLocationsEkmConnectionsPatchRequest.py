from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsEkmConnectionsPatchRequest(_messages.Message):
    """A CloudkmsProjectsLocationsEkmConnectionsPatchRequest object.

  Fields:
    ekmConnection: A EkmConnection resource to be passed as the request body.
    name: Output only. The resource name for the EkmConnection in the format
      `projects/*/locations/*/ekmConnections/*`.
    updateMask: Required. List of fields to be updated in this request.
  """
    ekmConnection = _messages.MessageField('EkmConnection', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)