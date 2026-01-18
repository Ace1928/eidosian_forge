from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CidrBlock(_messages.Message):
    """CidrBlock contains an optional name and one CIDR block.

  Fields:
    cidrBlock: Optional. cidr_block must be specified in CIDR notation when
      using master_authorized_networks_config. Currently, the user could still
      use the deprecated man_block field, so this field is currently optional,
      but will be required in the future.
    displayName: Optional. display_name is an optional field for users to
      identify CIDR blocks.
  """
    cidrBlock = _messages.StringField(1)
    displayName = _messages.StringField(2)