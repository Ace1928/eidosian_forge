from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1InstanceDeploymentStatus(_messages.Message):
    """The status of a deployment as reported by a single instance.

  Fields:
    deployedRevisions: Revisions currently deployed in MPs.
    deployedRoutes: Current routes deployed in the ingress routing table. A
      route which is missing will appear in `missing_routes`.
    instance: ID of the instance reporting the status.
  """
    deployedRevisions = _messages.MessageField('GoogleCloudApigeeV1InstanceDeploymentStatusDeployedRevision', 1, repeated=True)
    deployedRoutes = _messages.MessageField('GoogleCloudApigeeV1InstanceDeploymentStatusDeployedRoute', 2, repeated=True)
    instance = _messages.StringField(3)