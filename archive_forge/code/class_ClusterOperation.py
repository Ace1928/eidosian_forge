from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterOperation(_messages.Message):
    """The cluster operation triggered by a workflow.

  Fields:
    done: Output only. Indicates the operation is done.
    error: Output only. Error, if operation failed.
    operationId: Output only. The id of the cluster operation.
  """
    done = _messages.BooleanField(1)
    error = _messages.StringField(2)
    operationId = _messages.StringField(3)