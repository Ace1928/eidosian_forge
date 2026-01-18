from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsPatchRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydrat
  edDeploymentsPatchRequest object.

  Fields:
    hydratedDeployment: A HydratedDeployment resource to be passed as the
      request body.
    name: Output only. The name of the hydrated deployment.
    updateMask: Required. The list of fields to update. Update mask supports a
      special value `*` which fully replaces (equivalent to PUT) the resource
      provided.
  """
    hydratedDeployment = _messages.MessageField('HydratedDeployment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)