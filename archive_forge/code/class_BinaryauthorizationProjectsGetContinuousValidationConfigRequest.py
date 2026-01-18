from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BinaryauthorizationProjectsGetContinuousValidationConfigRequest(_messages.Message):
    """A BinaryauthorizationProjectsGetContinuousValidationConfigRequest
  object.

  Fields:
    name: Required. The name of the continuous validation config.
  """
    name = _messages.StringField(1, required=True)