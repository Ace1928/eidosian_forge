from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowsProjectsLocationsWorkflowsListRevisionsRequest(_messages.Message):
    """A WorkflowsProjectsLocationsWorkflowsListRevisionsRequest object.

  Fields:
    name: Required. Workflow for which the revisions should be listed. Format:
      projects/{project}/locations/{location}/workflows/{workflow}
    pageSize: The maximum number of revisions to return per page. If a value
      is not specified, a default value of 20 is used. The maximum permitted
      value is 100. Values greater than 100 are coerced down to 100.
    pageToken: The page token, received from a previous ListWorkflowRevisions
      call. Provide this to retrieve the subsequent page.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)