from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigTableIODetails(_messages.Message):
    """Metadata for a Cloud Bigtable connector used by the job.

  Fields:
    instanceId: InstanceId accessed in the connection.
    projectId: ProjectId accessed in the connection.
    tableId: TableId accessed in the connection.
  """
    instanceId = _messages.StringField(1)
    projectId = _messages.StringField(2)
    tableId = _messages.StringField(3)