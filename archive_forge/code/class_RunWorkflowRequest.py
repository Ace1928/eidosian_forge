from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunWorkflowRequest(_messages.Message):
    """Message for running a Workflow.

  Fields:
    etag: Needed for declarative-friendly resources.
    params: Run-time params.
    validateOnly: When true, the query is validated only, but not executed.
  """
    etag = _messages.StringField(1)
    params = _messages.MessageField('Param', 2, repeated=True)
    validateOnly = _messages.BooleanField(3)