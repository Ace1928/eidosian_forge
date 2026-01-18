from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsEkmConnectionsCreateRequest(_messages.Message):
    """A CloudkmsProjectsLocationsEkmConnectionsCreateRequest object.

  Fields:
    ekmConnection: A EkmConnection resource to be passed as the request body.
    ekmConnectionId: Required. It must be unique within a location and match
      the regular expression `[a-zA-Z0-9_-]{1,63}`.
    parent: Required. The resource name of the location associated with the
      EkmConnection, in the format `projects/*/locations/*`.
  """
    ekmConnection = _messages.MessageField('EkmConnection', 1)
    ekmConnectionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)