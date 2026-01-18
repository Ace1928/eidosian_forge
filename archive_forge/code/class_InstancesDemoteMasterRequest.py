from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesDemoteMasterRequest(_messages.Message):
    """Database demote primary instance request.

  Fields:
    demoteMasterContext: Contains details about the demoteMaster operation.
  """
    demoteMasterContext = _messages.MessageField('DemoteMasterContext', 1)