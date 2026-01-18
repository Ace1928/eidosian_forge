from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GsuiteaddonsProjectsDeploymentsReplaceDeploymentRequest(_messages.Message):
    """A GsuiteaddonsProjectsDeploymentsReplaceDeploymentRequest object.

  Fields:
    googleCloudGsuiteaddonsV1Deployment: A GoogleCloudGsuiteaddonsV1Deployment
      resource to be passed as the request body.
    name: The deployment resource name. Example:
      `projects/123/deployments/my_deployment`.
  """
    googleCloudGsuiteaddonsV1Deployment = _messages.MessageField('GoogleCloudGsuiteaddonsV1Deployment', 1)
    name = _messages.StringField(2, required=True)