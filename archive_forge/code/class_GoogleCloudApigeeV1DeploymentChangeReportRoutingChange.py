from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeploymentChangeReportRoutingChange(_messages.Message):
    """Describes a potential routing change that may occur as a result of some
  deployment operation.

  Fields:
    description: Human-readable description of this routing change.
    environmentGroup: Name of the environment group affected by this routing
      change.
    fromDeployment: Base path/deployment that may stop receiving some traffic.
    shouldSequenceRollout: Set to `true` if using sequenced rollout would make
      this routing change safer. **Note**: This does not necessarily imply
      that automated sequenced rollout mode is supported for the operation.
    toDeployment: Base path/deployment that may start receiving that traffic.
      May be null if no deployment is able to receive the traffic.
  """
    description = _messages.StringField(1)
    environmentGroup = _messages.StringField(2)
    fromDeployment = _messages.MessageField('GoogleCloudApigeeV1DeploymentChangeReportRoutingDeployment', 3)
    shouldSequenceRollout = _messages.BooleanField(4)
    toDeployment = _messages.MessageField('GoogleCloudApigeeV1DeploymentChangeReportRoutingDeployment', 5)