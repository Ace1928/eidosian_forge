from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DiscoveryApisGetRestRequest(_messages.Message):
    """A DiscoveryApisGetRestRequest object.

  Fields:
    api: The name of the API.
    version: The version of the API.
  """
    api = _messages.StringField(1, required=True)
    version = _messages.StringField(2, required=True)