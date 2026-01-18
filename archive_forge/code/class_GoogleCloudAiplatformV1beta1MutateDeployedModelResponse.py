from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MutateDeployedModelResponse(_messages.Message):
    """Response message for EndpointService.MutateDeployedModel.

  Fields:
    deployedModel: The DeployedModel that's being mutated.
  """
    deployedModel = _messages.MessageField('GoogleCloudAiplatformV1beta1DeployedModel', 1)