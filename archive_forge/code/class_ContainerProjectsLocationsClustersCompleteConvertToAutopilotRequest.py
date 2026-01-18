from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsLocationsClustersCompleteConvertToAutopilotRequest(_messages.Message):
    """A ContainerProjectsLocationsClustersCompleteConvertToAutopilotRequest
  object.

  Fields:
    completeConvertToAutopilotRequest: A CompleteConvertToAutopilotRequest
      resource to be passed as the request body.
    name: The name (project, location, cluster) of the cluster to convert.
      Specified in the format `projects/*/locations/*/clusters/*`.
  """
    completeConvertToAutopilotRequest = _messages.MessageField('CompleteConvertToAutopilotRequest', 1)
    name = _messages.StringField(2, required=True)