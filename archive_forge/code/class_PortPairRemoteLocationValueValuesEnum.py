from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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