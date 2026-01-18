from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelCallToActionDeployGke(_messages.Message):
    """Configurations for PublisherModel GKE deployment

  Fields:
    gkeYamlConfigs: Optional. GKE deployment configuration in yaml format.
  """
    gkeYamlConfigs = _messages.StringField(1, repeated=True)