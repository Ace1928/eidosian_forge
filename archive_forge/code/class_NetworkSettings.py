from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkSettings(_messages.Message):
    """A NetworkSettings resource is a container for ingress settings for a
  version or service.

  Enums:
    IngressTrafficAllowedValueValuesEnum: The ingress settings for version or
      service.

  Fields:
    ingressTrafficAllowed: The ingress settings for version or service.
  """

    class IngressTrafficAllowedValueValuesEnum(_messages.Enum):
        """The ingress settings for version or service.

    Values:
      INGRESS_TRAFFIC_ALLOWED_UNSPECIFIED: Unspecified
      INGRESS_TRAFFIC_ALLOWED_ALL: Allow HTTP traffic from public and private
        sources.
      INGRESS_TRAFFIC_ALLOWED_INTERNAL_ONLY: Allow HTTP traffic from only
        private VPC sources.
      INGRESS_TRAFFIC_ALLOWED_INTERNAL_AND_LB: Allow HTTP traffic from private
        VPC sources and through load balancers.
    """
        INGRESS_TRAFFIC_ALLOWED_UNSPECIFIED = 0
        INGRESS_TRAFFIC_ALLOWED_ALL = 1
        INGRESS_TRAFFIC_ALLOWED_INTERNAL_ONLY = 2
        INGRESS_TRAFFIC_ALLOWED_INTERNAL_AND_LB = 3
    ingressTrafficAllowed = _messages.EnumField('IngressTrafficAllowedValueValuesEnum', 1)