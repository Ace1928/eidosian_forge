from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsDeploymentsRollbackRequest(_messages.Message):
    """A DataflowProjectsLocationsDeploymentsRollbackRequest object.

  Fields:
    name: Required. The name of the `Deployment`. Format:
      projects/{project}/locations/{location}/deployments/{deployment_id}
    rollbackDeploymentRequest: A RollbackDeploymentRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    rollbackDeploymentRequest = _messages.MessageField('RollbackDeploymentRequest', 2)