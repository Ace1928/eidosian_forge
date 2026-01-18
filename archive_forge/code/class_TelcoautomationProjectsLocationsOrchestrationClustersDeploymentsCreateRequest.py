from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsCreateRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsCreate
  Request object.

  Fields:
    deployment: A Deployment resource to be passed as the request body.
    deploymentId: Optional. The name of the deployment.
    parent: Required. The name of parent resource. Format should be - "project
      s/{project_id}/locations/{location_name}/orchestrationClusters/{orchestr
      ation_cluster}".
  """
    deployment = _messages.MessageField('Deployment', 1)
    deploymentId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)