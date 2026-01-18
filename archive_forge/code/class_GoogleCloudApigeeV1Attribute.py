from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Attribute(_messages.Message):
    """Key-value pair to store extra metadata.

  Fields:
    name: API key of the attribute.
    value: Value of the attribute.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)