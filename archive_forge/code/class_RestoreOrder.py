from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreOrder(_messages.Message):
    """Allows customers to specify dependencies between resources that Backup
  for GKE can use to compute a resasonable restore order.

  Fields:
    groupKindDependencies: Optional. Contains a list of group kind dependency
      pairs provided by the customer, that is used by Backup for GKE to
      generate a group kind restore order.
  """
    groupKindDependencies = _messages.MessageField('GroupKindDependency', 1, repeated=True)