from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowStatusUpdateOptions(_messages.Message):
    """Configure how/where status is posted.

  Enums:
    RepositoryStatusValueValuesEnum: Options that specify additional
      information related to a Repo that should be sent in Pub/Sub
      Notifications

  Fields:
    pubsubTopic: Controls which Pub/Sub topic is used to send status updates
      as a build progresses and terminates. Default: projects//pub-
      sub/topics/cloud-build
    repositoryStatus: Options that specify additional information related to a
      Repo that should be sent in Pub/Sub Notifications
  """

    class RepositoryStatusValueValuesEnum(_messages.Enum):
        """Options that specify additional information related to a Repo that
    should be sent in Pub/Sub Notifications

    Values:
      REPOSITORY_STATUS_UNSPECIFIED: Default value. This value is unused.
      REPOSITORY_STATUS_NAME: Include the event_source of the WorkflowTrigger
        that results in the PipelineRun/TaskRun
      REPOSITORY_STATUS_NAME_LOG: Include the GCL log url of the
        PipelineRun/TaskRun in addition to the event source
    """
        REPOSITORY_STATUS_UNSPECIFIED = 0
        REPOSITORY_STATUS_NAME = 1
        REPOSITORY_STATUS_NAME_LOG = 2
    pubsubTopic = _messages.StringField(1)
    repositoryStatus = _messages.EnumField('RepositoryStatusValueValuesEnum', 2)