from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkItem(_messages.Message):
    """WorkItem represents basic information about a WorkItem to be executed in
  the cloud.

  Fields:
    configuration: Work item-specific configuration as an opaque blob.
    id: Identifies this WorkItem.
    initialReportIndex: The initial index to use when reporting the status of
      the WorkItem.
    jobId: Identifies the workflow job this WorkItem belongs to.
    leaseExpireTime: Time when the lease on this Work will expire.
    mapTask: Additional information for MapTask WorkItems.
    packages: Any required packages that need to be fetched in order to
      execute this WorkItem.
    projectId: Identifies the cloud project this WorkItem belongs to.
    reportStatusInterval: Recommended reporting interval.
    seqMapTask: Additional information for SeqMapTask WorkItems.
    shellTask: Additional information for ShellTask WorkItems.
    sourceOperationTask: Additional information for source operation
      WorkItems.
    streamingComputationTask: Additional information for
      StreamingComputationTask WorkItems.
    streamingConfigTask: Additional information for StreamingConfigTask
      WorkItems.
    streamingSetupTask: Additional information for StreamingSetupTask
      WorkItems.
  """
    configuration = _messages.StringField(1)
    id = _messages.IntegerField(2)
    initialReportIndex = _messages.IntegerField(3)
    jobId = _messages.StringField(4)
    leaseExpireTime = _messages.StringField(5)
    mapTask = _messages.MessageField('MapTask', 6)
    packages = _messages.MessageField('Package', 7, repeated=True)
    projectId = _messages.StringField(8)
    reportStatusInterval = _messages.StringField(9)
    seqMapTask = _messages.MessageField('SeqMapTask', 10)
    shellTask = _messages.MessageField('ShellTask', 11)
    sourceOperationTask = _messages.MessageField('SourceOperationRequest', 12)
    streamingComputationTask = _messages.MessageField('StreamingComputationTask', 13)
    streamingConfigTask = _messages.MessageField('StreamingConfigTask', 14)
    streamingSetupTask = _messages.MessageField('StreamingSetupTask', 15)