from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesPingsourcesCreateRequest(_messages.Message):
    """A AnthoseventsNamespacesPingsourcesCreateRequest object.

  Fields:
    parent: The namespace name.
    pingSource: A PingSource resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    pingSource = _messages.MessageField('PingSource', 2)