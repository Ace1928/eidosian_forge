from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeployedModelRef(_messages.Message):
    """Points to a DeployedModel.

  Fields:
    deployedModelId: Immutable. An ID of a DeployedModel in the above
      Endpoint.
    endpoint: Immutable. A resource name of an Endpoint.
  """
    deployedModelId = _messages.StringField(1)
    endpoint = _messages.StringField(2)