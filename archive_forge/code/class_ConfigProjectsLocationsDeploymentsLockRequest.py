from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsDeploymentsLockRequest(_messages.Message):
    """A ConfigProjectsLocationsDeploymentsLockRequest object.

  Fields:
    lockDeploymentRequest: A LockDeploymentRequest resource to be passed as
      the request body.
    name: Required. The name of the deployment in the format:
      'projects/{project_id}/locations/{location}/deployments/{deployment}'.
  """
    lockDeploymentRequest = _messages.MessageField('LockDeploymentRequest', 1)
    name = _messages.StringField(2, required=True)