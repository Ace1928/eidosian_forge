from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesDemoteRequest(_messages.Message):
    """This request is used to demote an existing standalone instance to be a
  Cloud SQL read replica for an external database server.

  Fields:
    demoteContext: Required. This context is used to demote an existing
      standalone instance to be a Cloud SQL read replica for an external
      database server.
  """
    demoteContext = _messages.MessageField('DemoteContext', 1)