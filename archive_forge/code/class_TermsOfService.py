from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TermsOfService(_messages.Message):
    """TermsOfService captures the metadata about a given terms of service.

  Fields:
    displayName: Display name of the terms of service.
    url: URL at which the terms of service can be viewed.
  """
    displayName = _messages.StringField(1)
    url = _messages.StringField(2)