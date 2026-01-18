from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateNetworkMatcher(client, args):
    """Returns a SecurityPolicyRuleNetworkMatcher message."""
    messages = client.messages
    network_matcher = messages.SecurityPolicyRuleNetworkMatcher()
    update_mask = []
    is_updated = False
    if getattr(args, 'network_user_defined_fields', None) is not None:
        user_defined_fields = []
        for user_defined_field in args.network_user_defined_fields:
            parsed = user_defined_field.split(';')
            name = parsed[0]
            values = parsed[1].split(':')
            user_defined_fields.append(messages.SecurityPolicyRuleNetworkMatcherUserDefinedFieldMatch(name=name, values=values))
        network_matcher.userDefinedFields = user_defined_fields
        update_mask.append('network_match.user_defined_fields')
        is_updated = True
    if getattr(args, 'network_src_ip_ranges', None) is not None:
        network_matcher.srcIpRanges = args.network_src_ip_ranges
        update_mask.append('network_match.src_ip_ranges')
        is_updated = True
    if getattr(args, 'network_dest_ip_ranges', None) is not None:
        network_matcher.destIpRanges = args.network_dest_ip_ranges
        update_mask.append('network_match.dest_ip_ranges')
        is_updated = True
    if getattr(args, 'network_ip_protocols', None) is not None:
        network_matcher.ipProtocols = args.network_ip_protocols
        update_mask.append('network_match.ip_protocols')
        is_updated = True
    if getattr(args, 'network_src_ports', None) is not None:
        network_matcher.srcPorts = args.network_src_ports
        update_mask.append('network_match.src_ports')
        is_updated = True
    if getattr(args, 'network_dest_ports', None) is not None:
        network_matcher.destPorts = args.network_dest_ports
        update_mask.append('network_match.dest_ports')
        is_updated = True
    if getattr(args, 'network_src_region_codes', None) is not None:
        network_matcher.srcRegionCodes = args.network_src_region_codes
        update_mask.append('network_match.src_region_codes')
        is_updated = True
    if getattr(args, 'network_src_asns', None) is not None:
        network_matcher.srcAsns = [int(asn) for asn in args.network_src_asns]
        update_mask.append('network_match.src_asns')
        is_updated = True
    update_mask_str = ','.join(update_mask)
    return (network_matcher, update_mask_str) if is_updated else (None, None)