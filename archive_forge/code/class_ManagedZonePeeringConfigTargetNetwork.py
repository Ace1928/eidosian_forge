from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZonePeeringConfigTargetNetwork(_messages.Message):
    """A ManagedZonePeeringConfigTargetNetwork object.

  Fields:
    deactivateTime: The time at which the zone was deactivated, in RFC 3339
      date-time format. An empty string indicates that the peering connection
      is active. The producer network can deactivate a zone. The zone is
      automatically deactivated if the producer network that the zone targeted
      is deleted. Output only.
    kind: A string attribute.
    networkUrl: The fully qualified URL of the VPC network to forward queries
      to. This should be formatted like https://www.googleapis.com/compute/v1/
      projects/{project}/global/networks/{network}
  """
    deactivateTime = _messages.StringField(1)
    kind = _messages.StringField(2, default='dns#managedZonePeeringConfigTargetNetwork')
    networkUrl = _messages.StringField(3)