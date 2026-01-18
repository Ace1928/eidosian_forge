from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeeringValueValuesEnum(_messages.Enum):
    """The type of peering set for this internal range.

    Values:
      PEERING_UNSPECIFIED: If Peering is left unspecified in
        CreateInternalRange or UpdateInternalRange, it will be defaulted to
        FOR_SELF.
      FOR_SELF: This is the default behavior and represents the case that this
        internal range is intended to be used in the VPC in which it is
        created and is accessible from its peers. This implies that peers or
        peers-of-peers cannot use this range.
      FOR_PEER: This behavior can be set when the internal range is being
        reserved for usage by peers. This means that no resource within the
        VPC in which it is being created can use this to associate with a VPC
        resource, but one of the peers can. This represents donating a range
        for peers to use.
      NOT_SHARED: This behavior can be set when the internal range is being
        reserved for usage by the VPC in which it is created, but not shared
        with peers. In a sense, it is local to the VPC. This can be used to
        create internal ranges for various purposes like
        HTTP_INTERNAL_LOAD_BALANCER or for Interconnect routes that are not
        shared with peers. This also implies that peers cannot use this range
        in a way that is visible to this VPC, but can re-use this range as
        long as it is NOT_SHARED from the peer VPC, too.
    """
    PEERING_UNSPECIFIED = 0
    FOR_SELF = 1
    FOR_PEER = 2
    NOT_SHARED = 3