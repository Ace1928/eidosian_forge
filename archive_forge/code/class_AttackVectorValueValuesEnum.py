from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttackVectorValueValuesEnum(_messages.Enum):
    """This metric reflects the context by which vulnerability exploitation
    is possible.

    Values:
      ATTACK_VECTOR_UNSPECIFIED: Invalid value.
      ATTACK_VECTOR_NETWORK: The vulnerable component is bound to the network
        stack and the set of possible attackers extends beyond the other
        options listed below, up to and including the entire Internet.
      ATTACK_VECTOR_ADJACENT: The vulnerable component is bound to the network
        stack, but the attack is limited at the protocol level to a logically
        adjacent topology.
      ATTACK_VECTOR_LOCAL: The vulnerable component is not bound to the
        network stack and the attacker's path is via read/write/execute
        capabilities.
      ATTACK_VECTOR_PHYSICAL: The attack requires the attacker to physically
        touch or manipulate the vulnerable component.
    """
    ATTACK_VECTOR_UNSPECIFIED = 0
    ATTACK_VECTOR_NETWORK = 1
    ATTACK_VECTOR_ADJACENT = 2
    ATTACK_VECTOR_LOCAL = 3
    ATTACK_VECTOR_PHYSICAL = 4