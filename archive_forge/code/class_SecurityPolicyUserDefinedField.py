from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyUserDefinedField(_messages.Message):
    """A SecurityPolicyUserDefinedField object.

  Enums:
    BaseValueValuesEnum: The base relative to which 'offset' is measured.
      Possible values are: - IPV4: Points to the beginning of the IPv4 header.
      - IPV6: Points to the beginning of the IPv6 header. - TCP: Points to the
      beginning of the TCP header, skipping over any IPv4 options or IPv6
      extension headers. Not present for non-first fragments. - UDP: Points to
      the beginning of the UDP header, skipping over any IPv4 options or IPv6
      extension headers. Not present for non-first fragments. required

  Fields:
    base: The base relative to which 'offset' is measured. Possible values
      are: - IPV4: Points to the beginning of the IPv4 header. - IPV6: Points
      to the beginning of the IPv6 header. - TCP: Points to the beginning of
      the TCP header, skipping over any IPv4 options or IPv6 extension
      headers. Not present for non-first fragments. - UDP: Points to the
      beginning of the UDP header, skipping over any IPv4 options or IPv6
      extension headers. Not present for non-first fragments. required
    mask: If specified, apply this mask (bitwise AND) to the field to ignore
      bits before matching. Encoded as a hexadecimal number (starting with
      "0x"). The last byte of the field (in network byte order) corresponds to
      the least significant byte of the mask.
    name: The name of this field. Must be unique within the policy.
    offset: Offset of the first byte of the field (in network byte order)
      relative to 'base'.
    size: Size of the field in bytes. Valid values: 1-4.
  """

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
    base = _messages.EnumField('BaseValueValuesEnum', 1)
    mask = _messages.StringField(2)
    name = _messages.StringField(3)
    offset = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    size = _messages.IntegerField(5, variant=_messages.Variant.INT32)