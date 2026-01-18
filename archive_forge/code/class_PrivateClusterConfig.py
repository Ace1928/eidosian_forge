from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateClusterConfig(_messages.Message):
    """Configuration options for the private GKE cluster in a Cloud Composer
  environment.

  Fields:
    enablePrivateEndpoint: Optional. If `true`, access to the public endpoint
      of the GKE cluster is denied.
    masterIpv4CidrBlock: Optional. The CIDR block from which IPv4 range for
      GKE master will be reserved. If left blank, the default value of
      '172.16.0.0/23' is used.
    masterIpv4ReservedRange: Output only. The IP range in CIDR notation to use
      for the hosted master network. This range is used for assigning internal
      IP addresses to the GKE cluster master or set of masters and to the
      internal load balancer virtual IP. This range must not overlap with any
      other ranges in use within the cluster's network.
  """
    enablePrivateEndpoint = _messages.BooleanField(1)
    masterIpv4CidrBlock = _messages.StringField(2)
    masterIpv4ReservedRange = _messages.StringField(3)