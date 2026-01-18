from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListManagedZonesResponse(_messages.Message):
    """Response message for ConnectorsService.ListManagedZones

  Fields:
    managedZones: ManagedZones.
    nextPageToken: Next page token.
  """
    managedZones = _messages.MessageField('ManagedZone', 1, repeated=True)
    nextPageToken = _messages.StringField(2)