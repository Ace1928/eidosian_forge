from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerDeploymentsCancelPreviewRequest(_messages.Message):
    """A DeploymentmanagerDeploymentsCancelPreviewRequest object.

  Fields:
    deployment: The name of the deployment for this request.
    deploymentsCancelPreviewRequest: A DeploymentsCancelPreviewRequest
      resource to be passed as the request body.
    project: The project ID for this request.
  """
    deployment = _messages.StringField(1, required=True)
    deploymentsCancelPreviewRequest = _messages.MessageField('DeploymentsCancelPreviewRequest', 2)
    project = _messages.StringField(3, required=True)