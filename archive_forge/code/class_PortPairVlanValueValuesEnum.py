from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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