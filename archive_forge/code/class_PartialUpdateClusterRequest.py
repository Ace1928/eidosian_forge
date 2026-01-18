from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartialUpdateClusterRequest(_messages.Message):
    """Request message for BigtableInstanceAdmin.PartialUpdateCluster.

  Fields:
    cluster: Required. The Cluster which contains the partial updates to be
      applied, subject to the update_mask.
    updateMask: Required. The subset of Cluster fields which should be
      replaced.
  """
    cluster = _messages.MessageField('Cluster', 1)
    updateMask = _messages.StringField(2)