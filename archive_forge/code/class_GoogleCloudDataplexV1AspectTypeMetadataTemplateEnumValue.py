from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AspectTypeMetadataTemplateEnumValue(_messages.Message):
    """Definition of Enumvalue (to be used by enum fields)

  Fields:
    deprecated: Optional. Optional deprecation message to be set if an enum
      value needs to be deprecated.
    index: Required. Index for the enum. Cannot be modified.
    name: Required. Name of the enumvalue. This is the actual value that the
      aspect will contain.
  """
    deprecated = _messages.StringField(1)
    index = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    name = _messages.StringField(3)