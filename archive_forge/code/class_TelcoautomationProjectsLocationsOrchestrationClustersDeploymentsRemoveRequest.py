from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsRemoveRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsRemove
  Request object.

  Fields:
    name: Required. The name of deployment to initiate delete.
    removeDeploymentRequest: A RemoveDeploymentRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    removeDeploymentRequest = _messages.MessageField('RemoveDeploymentRequest', 2)