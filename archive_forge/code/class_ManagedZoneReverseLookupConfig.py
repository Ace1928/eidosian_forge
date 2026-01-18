from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZoneReverseLookupConfig(_messages.Message):
    """A ManagedZoneReverseLookupConfig object.

  Fields:
    kind: A string attribute.
  """
    kind = _messages.StringField(1, default='dns#managedZoneReverseLookupConfig')