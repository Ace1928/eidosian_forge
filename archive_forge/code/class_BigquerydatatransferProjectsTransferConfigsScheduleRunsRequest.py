from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsTransferConfigsScheduleRunsRequest(_messages.Message):
    """A BigquerydatatransferProjectsTransferConfigsScheduleRunsRequest object.

  Fields:
    parent: Required. Transfer configuration name in the form:
      `projects/{project_id}/transferConfigs/{config_id}` or `projects/{projec
      t_id}/locations/{location_id}/transferConfigs/{config_id}`.
    scheduleTransferRunsRequest: A ScheduleTransferRunsRequest resource to be
      passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    scheduleTransferRunsRequest = _messages.MessageField('ScheduleTransferRunsRequest', 2)