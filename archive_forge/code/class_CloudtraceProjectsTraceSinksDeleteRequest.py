from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudtraceProjectsTraceSinksDeleteRequest(_messages.Message):
    """A CloudtraceProjectsTraceSinksDeleteRequest object.

  Fields:
    name: Required. The full resource name of the sink to delete, including
      the parent resource and the sink identifier:
      "projects/[PROJECT_NUMBER]/traceSinks/[SINK_ID]" Example:
      `"projects/12345/traceSinks/my-sink-id"`.
  """
    name = _messages.StringField(1, required=True)