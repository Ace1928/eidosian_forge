from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsComputeDeploymentStatusRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsComput
  eDeploymentStatusRequest object.

  Fields:
    name: Required. The name of the deployment without revisionID.
  """
    name = _messages.StringField(1, required=True)