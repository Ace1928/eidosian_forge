from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetClusterSpec(_messages.Message):
    """Message describing TargetClusterSpec object

  Fields:
    membership: Required. gkehub membership of target cluster
    variant: variant to be synced
  """
    membership = _messages.StringField(1)
    variant = _messages.StringField(2)