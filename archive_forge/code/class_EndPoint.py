from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndPoint(_messages.Message):
    """Endpoint message includes details of the Destination endpoint.

  Fields:
    endpointUri: The URI of the Endpoint.
    headers: List of Header to be added to the Endpoint.
  """
    endpointUri = _messages.StringField(1)
    headers = _messages.MessageField('Header', 2, repeated=True)