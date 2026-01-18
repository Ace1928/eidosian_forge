from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class DnsManagedZonesGetRequest(_messages.Message):
    """A DnsManagedZonesGetRequest object.

  Fields:
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or id.
    project: Identifies the project addressed by this request.
  """
    managedZone = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)