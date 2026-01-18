from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeActionSchema(_messages.Message):
    """Schema of a runtime action.

  Fields:
    action: Output only. Name of the action.
    inputParameters: Output only. List of input parameter metadata for the
      action.
    resultMetadata: Output only. List of result field metadata.
  """
    action = _messages.StringField(1)
    inputParameters = _messages.MessageField('InputParameter', 2, repeated=True)
    resultMetadata = _messages.MessageField('ResultMetadata', 3, repeated=True)