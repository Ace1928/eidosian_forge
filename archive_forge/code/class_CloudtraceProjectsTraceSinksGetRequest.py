from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudtraceProjectsTraceSinksGetRequest(_messages.Message):
    """A CloudtraceProjectsTraceSinksGetRequest object.

  Fields:
    name: Required. The resource name of the sink:
      "projects/[PROJECT_NUMBER]/traceSinks/[SINK_ID]" Example:
      `"projects/12345/traceSinks/my-sink-id"`.
  """
    name = _messages.StringField(1, required=True)