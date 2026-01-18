from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsDiscardRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsDiscar
  dRequest object.

  Fields:
    discardDeploymentChangesRequest: A DiscardDeploymentChangesRequest
      resource to be passed as the request body.
    name: Required. The name of the deployment of which changes are being
      discarded.
  """
    discardDeploymentChangesRequest = _messages.MessageField('DiscardDeploymentChangesRequest', 1)
    name = _messages.StringField(2, required=True)