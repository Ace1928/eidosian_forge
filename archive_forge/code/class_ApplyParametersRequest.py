from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyParametersRequest(_messages.Message):
    """Request for ApplyParameters.

  Fields:
    applyAll: Whether to apply instance-level parameter group to all nodes. If
      set to true, users are restricted from specifying individual nodes, and
      `ApplyParameters` updates all nodes within the instance.
    nodeIds: Nodes to which the instance-level parameter group is applied.
  """
    applyAll = _messages.BooleanField(1)
    nodeIds = _messages.StringField(2, repeated=True)