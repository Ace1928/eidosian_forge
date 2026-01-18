from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DesiredStackTypeValueValuesEnum(_messages.Enum):
    """The desired stack type of the cluster. If a stack type is provided and
    does not match the current stack type of the cluster, update will attempt
    to change the stack type to the new type.

    Values:
      STACK_TYPE_UNSPECIFIED: Default value, will be defaulted as IPV4 only
      IPV4: Cluster is IPV4 only
      IPV4_IPV6: Cluster can use both IPv4 and IPv6
    """
    STACK_TYPE_UNSPECIFIED = 0
    IPV4 = 1
    IPV4_IPV6 = 2