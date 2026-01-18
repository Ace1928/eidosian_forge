from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataFilterLabelMatch(_messages.Message):
    """MetadataFilter label name value pairs that are expected to match
  corresponding labels presented as metadata to the load balancer.

  Fields:
    name: Name of metadata label. The name can have a maximum length of 1024
      characters and must be at least 1 character long.
    value: The value of the label must match the specified value. value can
      have a maximum length of 1024 characters.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)