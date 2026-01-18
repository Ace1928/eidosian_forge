from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsApplyRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydrat
  edDeploymentsApplyRequest object.

  Fields:
    applyHydratedDeploymentRequest: A ApplyHydratedDeploymentRequest resource
      to be passed as the request body.
    name: Required. The name of the hydrated deployment to apply.
  """
    applyHydratedDeploymentRequest = _messages.MessageField('ApplyHydratedDeploymentRequest', 1)
    name = _messages.StringField(2, required=True)