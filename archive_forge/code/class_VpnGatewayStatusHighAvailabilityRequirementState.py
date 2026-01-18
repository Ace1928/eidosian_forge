from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpnGatewayStatusHighAvailabilityRequirementState(_messages.Message):
    """Describes the high availability requirement state for the VPN connection
  between this Cloud VPN gateway and a peer gateway.

  Enums:
    StateValueValuesEnum: Indicates the high availability requirement state
      for the VPN connection. Valid values are CONNECTION_REDUNDANCY_MET,
      CONNECTION_REDUNDANCY_NOT_MET.
    UnsatisfiedReasonValueValuesEnum: Indicates the reason why the VPN
      connection does not meet the high availability redundancy
      criteria/requirement. Valid values is INCOMPLETE_TUNNELS_COVERAGE.

  Fields:
    state: Indicates the high availability requirement state for the VPN
      connection. Valid values are CONNECTION_REDUNDANCY_MET,
      CONNECTION_REDUNDANCY_NOT_MET.
    unsatisfiedReason: Indicates the reason why the VPN connection does not
      meet the high availability redundancy criteria/requirement. Valid values
      is INCOMPLETE_TUNNELS_COVERAGE.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Indicates the high availability requirement state for the VPN
    connection. Valid values are CONNECTION_REDUNDANCY_MET,
    CONNECTION_REDUNDANCY_NOT_MET.

    Values:
      CONNECTION_REDUNDANCY_MET: VPN tunnels are configured with adequate
        redundancy from Cloud VPN gateway to the peer VPN gateway. For both
        GCP-to-non-GCP and GCP-to-GCP connections, the adequate redundancy is
        a pre-requirement for users to get 99.99% availability on GCP side;
        please note that for any connection, end-to-end 99.99% availability is
        subject to proper configuration on the peer VPN gateway.
      CONNECTION_REDUNDANCY_NOT_MET: VPN tunnels are not configured with
        adequate redundancy from the Cloud VPN gateway to the peer gateway
    """
        CONNECTION_REDUNDANCY_MET = 0
        CONNECTION_REDUNDANCY_NOT_MET = 1

    class UnsatisfiedReasonValueValuesEnum(_messages.Enum):
        """Indicates the reason why the VPN connection does not meet the high
    availability redundancy criteria/requirement. Valid values is
    INCOMPLETE_TUNNELS_COVERAGE.

    Values:
      INCOMPLETE_TUNNELS_COVERAGE: <no description>
    """
        INCOMPLETE_TUNNELS_COVERAGE = 0
    state = _messages.EnumField('StateValueValuesEnum', 1)
    unsatisfiedReason = _messages.EnumField('UnsatisfiedReasonValueValuesEnum', 2)