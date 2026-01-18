from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HTTPDelivery(_messages.Message):
    """HTTPDelivery is the delivery configuration for an HTTP notification.

  Fields:
    uri: The URI to which JSON-containing HTTP POST requests should be sent.
  """
    uri = _messages.StringField(1)