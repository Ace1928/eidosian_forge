from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1QueryDeployedModelsResponse(_messages.Message):
    """Response message for QueryDeployedModels method.

  Fields:
    deployedModelRefs: References to the DeployedModels that share the
      specified deploymentResourcePool.
    deployedModels: DEPRECATED Use deployed_model_refs instead.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    totalDeployedModelCount: The total number of DeployedModels on this
      DeploymentResourcePool.
    totalEndpointCount: The total number of Endpoints that have DeployedModels
      on this DeploymentResourcePool.
  """
    deployedModelRefs = _messages.MessageField('GoogleCloudAiplatformV1DeployedModelRef', 1, repeated=True)
    deployedModels = _messages.MessageField('GoogleCloudAiplatformV1DeployedModel', 2, repeated=True)
    nextPageToken = _messages.StringField(3)
    totalDeployedModelCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    totalEndpointCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)