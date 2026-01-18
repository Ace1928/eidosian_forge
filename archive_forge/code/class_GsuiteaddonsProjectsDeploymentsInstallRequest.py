from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GsuiteaddonsProjectsDeploymentsInstallRequest(_messages.Message):
    """A GsuiteaddonsProjectsDeploymentsInstallRequest object.

  Fields:
    googleCloudGsuiteaddonsV1InstallDeploymentRequest: A
      GoogleCloudGsuiteaddonsV1InstallDeploymentRequest resource to be passed
      as the request body.
    name: Required. The full resource name of the deployment to install.
      Example: `projects/my_project/deployments/my_deployment`.
  """
    googleCloudGsuiteaddonsV1InstallDeploymentRequest = _messages.MessageField('GoogleCloudGsuiteaddonsV1InstallDeploymentRequest', 1)
    name = _messages.StringField(2, required=True)