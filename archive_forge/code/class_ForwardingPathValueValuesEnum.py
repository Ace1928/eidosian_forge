from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ForwardingPathValueValuesEnum(_messages.Enum):
    """Forwarding path for this TargetNameServer. If unset or set to DEFAULT,
    Cloud DNS makes forwarding decisions based on address ranges; that is,
    RFC1918 addresses go to the VPC network, non-RFC1918 addresses go to the
    internet. When set to PRIVATE, Cloud DNS always sends queries through the
    VPC network for this target.

    Values:
      default: Cloud DNS makes forwarding decision based on IP address ranges;
        that is, RFC1918 addresses forward to the target through the VPC and
        non-RFC1918 addresses forward to the target through the internet
      private: Cloud DNS always forwards to this target through the VPC.
    """
    default = 0
    private = 1