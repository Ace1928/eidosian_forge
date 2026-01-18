from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuxiliaryNodeGroup(_messages.Message):
    """Node group identification and configuration information.

  Fields:
    nodeGroup: Required. Node group configuration.
    nodeGroupId: Optional. A node group ID. Generated if not specified.The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). Cannot begin or end with underscore or hyphen. Must
      consist of from 3 to 33 characters.
  """
    nodeGroup = _messages.MessageField('NodeGroup', 1)
    nodeGroupId = _messages.StringField(2)