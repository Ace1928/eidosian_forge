from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcNetworkScopeValueValuesEnum(_messages.Enum):
    """The scope of networks allowed to be associated with the firewall
    policy. This field can be either GLOBAL_VPC_NETWORK or
    REGIONAL_VPC_NETWORK. A firewall policy with the VPC scope set to
    GLOBAL_VPC_NETWORK is allowed to be attached only to global networks. When
    the VPC scope is set to REGIONAL_VPC_NETWORK the firewall policy is
    allowed to be attached only to regional networks in the same scope as the
    firewall policy. Note: if not specified then GLOBAL_VPC_NETWORK will be
    used.

    Values:
      GLOBAL_VPC_NETWORK: The firewall policy is allowed to be attached only
        to global networks.
      REGIONAL_VPC_NETWORK: The firewall policy is allowed to be attached only
        to regional networks in the same scope as the firewall policy. This
        option is applicable only to regional firewall policies.
    """
    GLOBAL_VPC_NETWORK = 0
    REGIONAL_VPC_NETWORK = 1