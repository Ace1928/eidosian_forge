from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaseValueValuesEnum(_messages.Enum):
    """The base relative to which 'offset' is measured. Possible values are:
    - IPV4: Points to the beginning of the IPv4 header. - IPV6: Points to the
    beginning of the IPv6 header. - TCP: Points to the beginning of the TCP
    header, skipping over any IPv4 options or IPv6 extension headers. Not
    present for non-first fragments. - UDP: Points to the beginning of the UDP
    header, skipping over any IPv4 options or IPv6 extension headers. Not
    present for non-first fragments. required

    Values:
      IPV4: <no description>
      IPV6: <no description>
      TCP: <no description>
      UDP: <no description>
    """
    IPV4 = 0
    IPV6 = 1
    TCP = 2
    UDP = 3