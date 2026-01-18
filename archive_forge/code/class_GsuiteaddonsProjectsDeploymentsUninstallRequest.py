from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GsuiteaddonsProjectsDeploymentsUninstallRequest(_messages.Message):
    """A GsuiteaddonsProjectsDeploymentsUninstallRequest object.

  Fields:
    googleCloudGsuiteaddonsV1UninstallDeploymentRequest: A
      GoogleCloudGsuiteaddonsV1UninstallDeploymentRequest resource to be
      passed as the request body.
    name: Required. The full resource name of the deployment to install.
      Example: `projects/my_project/deployments/my_deployment`.
  """
    googleCloudGsuiteaddonsV1UninstallDeploymentRequest = _messages.MessageField('GoogleCloudGsuiteaddonsV1UninstallDeploymentRequest', 1)
    name = _messages.StringField(2, required=True)