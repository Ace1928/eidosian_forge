from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudtraceProjectsTraceSinksCreateRequest(_messages.Message):
    """A CloudtraceProjectsTraceSinksCreateRequest object.

  Fields:
    parent: Required. The resource in which to create the sink (currently only
      project sinks are supported): "projects/[PROJECT_ID]" Examples:
      `"projects/my-trace-project"`, `"projects/123456789"`.
    traceSink: A TraceSink resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    traceSink = _messages.MessageField('TraceSink', 2)