from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportRuntimeEventRequest(_messages.Message):
    """Request for reporting a Managed Notebook Event.

  Fields:
    event: Required. The Event to be reported.
    vmId: Required. The VM hardware token for authenticating the VM.
      https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """
    event = _messages.MessageField('Event', 1)
    vmId = _messages.StringField(2)