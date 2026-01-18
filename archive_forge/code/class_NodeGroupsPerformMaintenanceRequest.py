from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupsPerformMaintenanceRequest(_messages.Message):
    """A NodeGroupsPerformMaintenanceRequest object.

  Fields:
    nodes: [Required] List of nodes affected by the call.
    startTime: The start time of the schedule. The timestamp is an RFC3339
      string.
  """
    nodes = _messages.StringField(1, repeated=True)
    startTime = _messages.StringField(2)