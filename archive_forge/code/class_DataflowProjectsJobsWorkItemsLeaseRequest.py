from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsWorkItemsLeaseRequest(_messages.Message):
    """A DataflowProjectsJobsWorkItemsLeaseRequest object.

  Fields:
    jobId: Identifies the workflow job this worker belongs to.
    leaseWorkItemRequest: A LeaseWorkItemRequest resource to be passed as the
      request body.
    projectId: Identifies the project this worker belongs to.
  """
    jobId = _messages.StringField(1, required=True)
    leaseWorkItemRequest = _messages.MessageField('LeaseWorkItemRequest', 2)
    projectId = _messages.StringField(3, required=True)