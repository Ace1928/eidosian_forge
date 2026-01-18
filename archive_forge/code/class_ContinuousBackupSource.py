from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContinuousBackupSource(_messages.Message):
    """Message describing a ContinuousBackupSource.

  Fields:
    cluster: Required. The source cluster from which to restore. This cluster
      must have continuous backup enabled for this operation to succeed. For
      the required format, see the comment on the Cluster.name field.
    pointInTime: Required. The point in time to restore to.
  """
    cluster = _messages.StringField(1)
    pointInTime = _messages.StringField(2)