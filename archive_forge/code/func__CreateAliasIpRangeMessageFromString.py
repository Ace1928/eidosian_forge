from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
def _CreateAliasIpRangeMessageFromString(messages, instance, alias_ip_range_string):
    """Returns a new AliasIpRange message by parsing the input string."""
    alias_ip_range = messages.AliasIpRange()
    tokens = alias_ip_range_string.split(':')
    if len(tokens) == 1:
        alias_ip_range.ipCidrRange = tokens[0]
    elif len(tokens) == 2:
        alias_ip_range.subnetworkRangeName = tokens[0]
        alias_ip_range.ipCidrRange = tokens[1]
    else:
        raise calliope_exceptions.InvalidArgumentException('aliases', _INVALID_FORMAT_MESSAGE_FOR_INSTANCE if instance else _INVALID_FORMAT_MESSAGE_FOR_INSTANCE_TEMPLATE)
    return alias_ip_range