from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesRotateServerCaRequest(_messages.Message):
    """Rotate Server CA request.

  Fields:
    rotateServerCaContext: Contains details about the rotate server CA
      operation.
  """
    rotateServerCaContext = _messages.MessageField('RotateServerCaContext', 1)