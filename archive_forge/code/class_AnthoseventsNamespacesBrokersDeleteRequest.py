from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesBrokersDeleteRequest(_messages.Message):
    """A AnthoseventsNamespacesBrokersDeleteRequest object.

  Fields:
    name: The relative name of the broker being deleted, including the
      namespace
  """
    name = _messages.StringField(1, required=True)