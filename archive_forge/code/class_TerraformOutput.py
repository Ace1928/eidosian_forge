from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerraformOutput(_messages.Message):
    """Describes a Terraform output.

  Fields:
    sensitive: Identifies whether Terraform has set this output as a potential
      sensitive value.
    value: Value of output.
  """
    sensitive = _messages.BooleanField(1)
    value = _messages.MessageField('extra_types.JsonValue', 2)