from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterNat(_messages.Message):
    """Represents a Nat resource. It enables the VMs within the specified
  subnetworks to access Internet without external IP addresses. It specifies a
  list of subnetworks (and the ranges within) that want to use NAT. Customers
  can also provide the external IPs that would be used for NAT. GCP would
  auto-allocate ephemeral IPs if no external IPs are provided.

  Enums:
    AutoNetworkTierValueValuesEnum: The network tier to use when automatically
      reserving NAT IP addresses. Must be one of: PREMIUM, STANDARD. If not
      specified, then the current project-level default tier is used.
    EndpointTypesValueListEntryValuesEnum:
    NatIpAllocateOptionValueValuesEnum: Specify the NatIpAllocateOption, which
      can take one of the following values: - MANUAL_ONLY: Uses only Nat IP
      addresses provided by customers. When there are not enough specified Nat
      IPs, the Nat service fails for new VMs. - AUTO_ONLY: Nat IPs are
      allocated by Google Cloud Platform; customers can't specify any Nat IPs.
      When choosing AUTO_ONLY, then nat_ip should be empty.
    SourceSubnetworkIpRangesToNatValueValuesEnum: Specify the Nat option,
      which can take one of the following values: -
      ALL_SUBNETWORKS_ALL_IP_RANGES: All of the IP ranges in every Subnetwork
      are allowed to Nat. - ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES: All of the
      primary IP ranges in every Subnetwork are allowed to Nat. -
      LIST_OF_SUBNETWORKS: A list of Subnetworks are allowed to Nat (specified
      in the field subnetwork below) The default is
      SUBNETWORK_IP_RANGE_TO_NAT_OPTION_UNSPECIFIED. Note that if this field
      contains ALL_SUBNETWORKS_ALL_IP_RANGES then there should not be any
      other Router.Nat section in any Router for this network in this region.
    TypeValueValuesEnum: Indicates whether this NAT is used for public or
      private IP translation. If unspecified, it defaults to PUBLIC.

  Fields:
    autoNetworkTier: The network tier to use when automatically reserving NAT
      IP addresses. Must be one of: PREMIUM, STANDARD. If not specified, then
      the current project-level default tier is used.
    drainNatIps: A list of URLs of the IP resources to be drained. These IPs
      must be valid static external IPs that have been assigned to the NAT.
      These IPs should be used for updating/patching a NAT only.
    enableDynamicPortAllocation: Enable Dynamic Port Allocation. If not
      specified, it is disabled by default. If set to true, - Dynamic Port
      Allocation will be enabled on this NAT config. -
      enableEndpointIndependentMapping cannot be set to true. - If minPorts is
      set, minPortsPerVm must be set to a power of two greater than or equal
      to 32. If minPortsPerVm is not set, a minimum of 32 ports will be
      allocated to a VM from this NAT config.
    enableEndpointIndependentMapping: A boolean attribute.
    endpointTypes: List of NAT-ted endpoint types supported by the Nat
      Gateway. If the list is empty, then it will be equivalent to include
      ENDPOINT_TYPE_VM
    icmpIdleTimeoutSec: Timeout (in seconds) for ICMP connections. Defaults to
      30s if not set.
    logConfig: Configure logging on this NAT.
    maxPortsPerVm: Maximum number of ports allocated to a VM from this NAT
      config when Dynamic Port Allocation is enabled. If Dynamic Port
      Allocation is not enabled, this field has no effect. If Dynamic Port
      Allocation is enabled, and this field is set, it must be set to a power
      of two greater than minPortsPerVm, or 64 if minPortsPerVm is not set. If
      Dynamic Port Allocation is enabled and this field is not set, a maximum
      of 65536 ports will be allocated to a VM from this NAT config.
    minPortsPerVm: Minimum number of ports allocated to a VM from this NAT
      config. If not set, a default number of ports is allocated to a VM. This
      is rounded up to the nearest power of 2. For example, if the value of
      this field is 50, at least 64 ports are allocated to a VM.
    name: Unique name of this Nat service. The name must be 1-63 characters
      long and comply with RFC1035.
    natIpAllocateOption: Specify the NatIpAllocateOption, which can take one
      of the following values: - MANUAL_ONLY: Uses only Nat IP addresses
      provided by customers. When there are not enough specified Nat IPs, the
      Nat service fails for new VMs. - AUTO_ONLY: Nat IPs are allocated by
      Google Cloud Platform; customers can't specify any Nat IPs. When
      choosing AUTO_ONLY, then nat_ip should be empty.
    natIps: A list of URLs of the IP resources used for this Nat service.
      These IP addresses must be valid static external IP addresses assigned
      to the project.
    rules: A list of rules associated with this NAT.
    sourceSubnetworkIpRangesToNat: Specify the Nat option, which can take one
      of the following values: - ALL_SUBNETWORKS_ALL_IP_RANGES: All of the IP
      ranges in every Subnetwork are allowed to Nat. -
      ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES: All of the primary IP ranges in
      every Subnetwork are allowed to Nat. - LIST_OF_SUBNETWORKS: A list of
      Subnetworks are allowed to Nat (specified in the field subnetwork below)
      The default is SUBNETWORK_IP_RANGE_TO_NAT_OPTION_UNSPECIFIED. Note that
      if this field contains ALL_SUBNETWORKS_ALL_IP_RANGES then there should
      not be any other Router.Nat section in any Router for this network in
      this region.
    subnetworks: A list of Subnetwork resources whose traffic should be
      translated by NAT Gateway. It is used only when LIST_OF_SUBNETWORKS is
      selected for the SubnetworkIpRangeToNatOption above.
    tcpEstablishedIdleTimeoutSec: Timeout (in seconds) for TCP established
      connections. Defaults to 1200s if not set.
    tcpTimeWaitTimeoutSec: Timeout (in seconds) for TCP connections that are
      in TIME_WAIT state. Defaults to 120s if not set.
    tcpTransitoryIdleTimeoutSec: Timeout (in seconds) for TCP transitory
      connections. Defaults to 30s if not set.
    type: Indicates whether this NAT is used for public or private IP
      translation. If unspecified, it defaults to PUBLIC.
    udpIdleTimeoutSec: Timeout (in seconds) for UDP connections. Defaults to
      30s if not set.
  """

    class AutoNetworkTierValueValuesEnum(_messages.Enum):
        """The network tier to use when automatically reserving NAT IP addresses.
    Must be one of: PREMIUM, STANDARD. If not specified, then the current
    project-level default tier is used.

    Values:
      FIXED_STANDARD: Public internet quality with fixed bandwidth.
      PREMIUM: High quality, Google-grade network tier, support for all
        networking products.
      STANDARD: Public internet quality, only limited support for other
        networking products.
      STANDARD_OVERRIDES_FIXED_STANDARD: (Output only) Temporary tier for
        FIXED_STANDARD when fixed standard tier is expired or not configured.
    """
        FIXED_STANDARD = 0
        PREMIUM = 1
        STANDARD = 2
        STANDARD_OVERRIDES_FIXED_STANDARD = 3

    class EndpointTypesValueListEntryValuesEnum(_messages.Enum):
        """EndpointTypesValueListEntryValuesEnum enum type.

    Values:
      ENDPOINT_TYPE_MANAGED_PROXY_LB: This is used for regional Application
        Load Balancers (internal and external) and regional proxy Network Load
        Balancers (internal and external) endpoints.
      ENDPOINT_TYPE_SWG: This is used for Secure Web Gateway endpoints.
      ENDPOINT_TYPE_VM: This is the default.
    """
        ENDPOINT_TYPE_MANAGED_PROXY_LB = 0
        ENDPOINT_TYPE_SWG = 1
        ENDPOINT_TYPE_VM = 2

    class NatIpAllocateOptionValueValuesEnum(_messages.Enum):
        """Specify the NatIpAllocateOption, which can take one of the following
    values: - MANUAL_ONLY: Uses only Nat IP addresses provided by customers.
    When there are not enough specified Nat IPs, the Nat service fails for new
    VMs. - AUTO_ONLY: Nat IPs are allocated by Google Cloud Platform;
    customers can't specify any Nat IPs. When choosing AUTO_ONLY, then nat_ip
    should be empty.

    Values:
      AUTO_ONLY: Nat IPs are allocated by GCP; customers can not specify any
        Nat IPs.
      MANUAL_ONLY: Only use Nat IPs provided by customers. When specified Nat
        IPs are not enough then the Nat service fails for new VMs.
    """
        AUTO_ONLY = 0
        MANUAL_ONLY = 1

    class SourceSubnetworkIpRangesToNatValueValuesEnum(_messages.Enum):
        """Specify the Nat option, which can take one of the following values: -
    ALL_SUBNETWORKS_ALL_IP_RANGES: All of the IP ranges in every Subnetwork
    are allowed to Nat. - ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES: All of the
    primary IP ranges in every Subnetwork are allowed to Nat. -
    LIST_OF_SUBNETWORKS: A list of Subnetworks are allowed to Nat (specified
    in the field subnetwork below) The default is
    SUBNETWORK_IP_RANGE_TO_NAT_OPTION_UNSPECIFIED. Note that if this field
    contains ALL_SUBNETWORKS_ALL_IP_RANGES then there should not be any other
    Router.Nat section in any Router for this network in this region.

    Values:
      ALL_SUBNETWORKS_ALL_IP_RANGES: All the IP ranges in every Subnetwork are
        allowed to Nat.
      ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES: All the primary IP ranges in
        every Subnetwork are allowed to Nat.
      LIST_OF_SUBNETWORKS: A list of Subnetworks are allowed to Nat (specified
        in the field subnetwork below)
    """
        ALL_SUBNETWORKS_ALL_IP_RANGES = 0
        ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES = 1
        LIST_OF_SUBNETWORKS = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Indicates whether this NAT is used for public or private IP
    translation. If unspecified, it defaults to PUBLIC.

    Values:
      PRIVATE: NAT used for private IP translation.
      PUBLIC: NAT used for public IP translation. This is the default.
    """
        PRIVATE = 0
        PUBLIC = 1
    autoNetworkTier = _messages.EnumField('AutoNetworkTierValueValuesEnum', 1)
    drainNatIps = _messages.StringField(2, repeated=True)
    enableDynamicPortAllocation = _messages.BooleanField(3)
    enableEndpointIndependentMapping = _messages.BooleanField(4)
    endpointTypes = _messages.EnumField('EndpointTypesValueListEntryValuesEnum', 5, repeated=True)
    icmpIdleTimeoutSec = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    logConfig = _messages.MessageField('RouterNatLogConfig', 7)
    maxPortsPerVm = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    minPortsPerVm = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    name = _messages.StringField(10)
    natIpAllocateOption = _messages.EnumField('NatIpAllocateOptionValueValuesEnum', 11)
    natIps = _messages.StringField(12, repeated=True)
    rules = _messages.MessageField('RouterNatRule', 13, repeated=True)
    sourceSubnetworkIpRangesToNat = _messages.EnumField('SourceSubnetworkIpRangesToNatValueValuesEnum', 14)
    subnetworks = _messages.MessageField('RouterNatSubnetworkToNat', 15, repeated=True)
    tcpEstablishedIdleTimeoutSec = _messages.IntegerField(16, variant=_messages.Variant.INT32)
    tcpTimeWaitTimeoutSec = _messages.IntegerField(17, variant=_messages.Variant.INT32)
    tcpTransitoryIdleTimeoutSec = _messages.IntegerField(18, variant=_messages.Variant.INT32)
    type = _messages.EnumField('TypeValueValuesEnum', 19)
    udpIdleTimeoutSec = _messages.IntegerField(20, variant=_messages.Variant.INT32)