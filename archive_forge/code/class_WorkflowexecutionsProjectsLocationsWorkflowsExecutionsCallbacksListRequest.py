from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCallbacksListRequest(_messages.Message):
    """A
  WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCallbacksListRequest
  object.

  Fields:
    pageSize: Maximum number of callbacks to return per call. The default
      value is 100 and is also the maximum value.
    pageToken: A page token, received from a previous `ListCallbacks` call.
      Provide this to retrieve the subsequent page. Note that pagination is
      applied to dynamic data. The list of callbacks returned can change
      between page requests if callbacks are created or deleted.
    parent: Required. Name of the execution for which the callbacks should be
      listed. Format: projects/{project}/locations/{location}/workflows/{workf
      low}/executions/{execution}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)