from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GsuiteaddonsProjectsDeploymentsCreateRequest(_messages.Message):
    """A GsuiteaddonsProjectsDeploymentsCreateRequest object.

  Fields:
    deploymentId: Required. The ID to use for this deployment. The full name
      of the created resource will be `projects//deployments/`.
    googleCloudGsuiteaddonsV1Deployment: A GoogleCloudGsuiteaddonsV1Deployment
      resource to be passed as the request body.
    parent: Required. Name of the project in which to create the deployment.
      Example: `projects/my_project`.
  """
    deploymentId = _messages.StringField(1)
    googleCloudGsuiteaddonsV1Deployment = _messages.MessageField('GoogleCloudGsuiteaddonsV1Deployment', 2)
    parent = _messages.StringField(3, required=True)