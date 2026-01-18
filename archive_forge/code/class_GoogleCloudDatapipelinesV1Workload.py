from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1Workload(_messages.Message):
    """Workload details for creating the pipeline jobs.

  Fields:
    dataflowFlexTemplateRequest: Template information and additional
      parameters needed to launch a Dataflow job using the flex launch API.
    dataflowLaunchTemplateRequest: Template information and additional
      parameters needed to launch a Dataflow job using the standard launch
      API.
  """
    dataflowFlexTemplateRequest = _messages.MessageField('GoogleCloudDatapipelinesV1LaunchFlexTemplateRequest', 1)
    dataflowLaunchTemplateRequest = _messages.MessageField('GoogleCloudDatapipelinesV1LaunchTemplateRequest', 2)