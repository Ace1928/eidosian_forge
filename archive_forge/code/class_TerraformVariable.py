from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerraformVariable(_messages.Message):
    """A Terraform input variable.

  Fields:
    inputValue: Input variable value.
  """
    inputValue = _messages.MessageField('extra_types.JsonValue', 1)