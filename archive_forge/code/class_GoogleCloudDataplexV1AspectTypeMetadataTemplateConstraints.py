from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AspectTypeMetadataTemplateConstraints(_messages.Message):
    """Definition of the constraints of a field

  Fields:
    required: Optional. Marks this as an optional/required field.
  """
    required = _messages.BooleanField(1)