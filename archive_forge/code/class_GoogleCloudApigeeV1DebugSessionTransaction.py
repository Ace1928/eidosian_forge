from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DebugSessionTransaction(_messages.Message):
    """A transaction contains all of the debug information of the entire
  message flow of an API call processed by the runtime plane. The information
  is collected and recorded at critical points of the message flow in the
  runtime apiproxy.

  Fields:
    completed: Flag indicating whether a transaction is completed or not
    point: List of debug data collected by runtime plane at various defined
      points in the flow.
  """
    completed = _messages.BooleanField(1)
    point = _messages.MessageField('GoogleCloudApigeeV1Point', 2, repeated=True)