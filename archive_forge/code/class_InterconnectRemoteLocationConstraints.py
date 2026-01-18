from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectRemoteLocationConstraints(_messages.Message):
    """A InterconnectRemoteLocationConstraints object.

  Enums:
    PortPairRemoteLocationValueValuesEnum: [Output Only] Port pair remote
      location constraints, which can take one of the following values:
      PORT_PAIR_UNCONSTRAINED_REMOTE_LOCATION,
      PORT_PAIR_MATCHING_REMOTE_LOCATION. Google Cloud API refers only to
      individual ports, but the UI uses this field when ordering a pair of
      ports, to prevent users from accidentally ordering something that is
      incompatible with their cloud provider. Specifically, when ordering a
      redundant pair of Cross-Cloud Interconnect ports, and one of them uses a
      remote location with portPairMatchingRemoteLocation set to matching, the
      UI requires that both ports use the same remote location.
    PortPairVlanValueValuesEnum: [Output Only] Port pair VLAN constraints,
      which can take one of the following values:
      PORT_PAIR_UNCONSTRAINED_VLAN, PORT_PAIR_MATCHING_VLAN

  Fields:
    portPairRemoteLocation: [Output Only] Port pair remote location
      constraints, which can take one of the following values:
      PORT_PAIR_UNCONSTRAINED_REMOTE_LOCATION,
      PORT_PAIR_MATCHING_REMOTE_LOCATION. Google Cloud API refers only to
      individual ports, but the UI uses this field when ordering a pair of
      ports, to prevent users from accidentally ordering something that is
      incompatible with their cloud provider. Specifically, when ordering a
      redundant pair of Cross-Cloud Interconnect ports, and one of them uses a
      remote location with portPairMatchingRemoteLocation set to matching, the
      UI requires that both ports use the same remote location.
    portPairVlan: [Output Only] Port pair VLAN constraints, which can take one
      of the following values: PORT_PAIR_UNCONSTRAINED_VLAN,
      PORT_PAIR_MATCHING_VLAN
    subnetLengthRange: [Output Only] [min-length, max-length] The minimum and
      maximum value (inclusive) for the IPv4 subnet length. For example, an
      interconnectRemoteLocation for Azure has {min: 30, max: 30} because
      Azure requires /30 subnets. This range specifies the values supported by
      both cloud providers. Interconnect currently supports /29 and /30 IPv4
      subnet lengths. If a remote cloud has no constraint on IPv4 subnet
      length, the range would thus be {min: 29, max: 30}.
  """

    class PortPairRemoteLocationValueValuesEnum(_messages.Enum):
        """[Output Only] Port pair remote location constraints, which can take
    one of the following values: PORT_PAIR_UNCONSTRAINED_REMOTE_LOCATION,
    PORT_PAIR_MATCHING_REMOTE_LOCATION. Google Cloud API refers only to
    individual ports, but the UI uses this field when ordering a pair of
    ports, to prevent users from accidentally ordering something that is
    incompatible with their cloud provider. Specifically, when ordering a
    redundant pair of Cross-Cloud Interconnect ports, and one of them uses a
    remote location with portPairMatchingRemoteLocation set to matching, the
    UI requires that both ports use the same remote location.

    Values:
      PORT_PAIR_MATCHING_REMOTE_LOCATION: If
        PORT_PAIR_MATCHING_REMOTE_LOCATION, the remote cloud provider
        allocates ports in pairs, and the user should choose the same remote
        location for both ports.
      PORT_PAIR_UNCONSTRAINED_REMOTE_LOCATION: If
        PORT_PAIR_UNCONSTRAINED_REMOTE_LOCATION, a user may opt to provision a
        redundant pair of Cross-Cloud Interconnects using two different remote
        locations in the same city.
    """
        PORT_PAIR_MATCHING_REMOTE_LOCATION = 0
        PORT_PAIR_UNCONSTRAINED_REMOTE_LOCATION = 1

    class PortPairVlanValueValuesEnum(_messages.Enum):
        """[Output Only] Port pair VLAN constraints, which can take one of the
    following values: PORT_PAIR_UNCONSTRAINED_VLAN, PORT_PAIR_MATCHING_VLAN

    Values:
      PORT_PAIR_MATCHING_VLAN: If PORT_PAIR_MATCHING_VLAN, the Interconnect
        for this attachment is part of a pair of ports that should have
        matching VLAN allocations. This occurs with Cross-Cloud Interconnect
        to Azure remote locations. While GCP's API does not explicitly group
        pairs of ports, the UI uses this field to ensure matching VLAN ids
        when configuring a redundant VLAN pair.
      PORT_PAIR_UNCONSTRAINED_VLAN: PORT_PAIR_UNCONSTRAINED_VLAN means there
        is no constraint.
    """
        PORT_PAIR_MATCHING_VLAN = 0
        PORT_PAIR_UNCONSTRAINED_VLAN = 1
    portPairRemoteLocation = _messages.EnumField('PortPairRemoteLocationValueValuesEnum', 1)
    portPairVlan = _messages.EnumField('PortPairVlanValueValuesEnum', 2)
    subnetLengthRange = _messages.MessageField('InterconnectRemoteLocationConstraintsSubnetLengthRange', 3)