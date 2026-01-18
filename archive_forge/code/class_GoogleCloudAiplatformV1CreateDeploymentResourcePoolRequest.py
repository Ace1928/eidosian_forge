from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1CreateDeploymentResourcePoolRequest(_messages.Message):
    """Request message for CreateDeploymentResourcePool method.

  Fields:
    deploymentResourcePool: Required. The DeploymentResourcePool to create.
    deploymentResourcePoolId: Required. The ID to use for the
      DeploymentResourcePool, which will become the final component of the
      DeploymentResourcePool's resource name. The maximum length is 63
      characters, and valid characters are
      `/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/`.
  """
    deploymentResourcePool = _messages.MessageField('GoogleCloudAiplatformV1DeploymentResourcePool', 1)
    deploymentResourcePoolId = _messages.StringField(2)