from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetResourceRecordSetsRoutingPolicyDataArg(required=False, deprecated_name=False):
    """Returns --routing-policy-data command line arg value."""

    def RoutingPolicyDataArgType(routing_policy_data_value):
        """Converts --routing-policy-data flag value to a list of policy data items.

    Args:
      routing_policy_data_value: String value specified in the
        --routing-policy-data flag.

    Returns:
      A list of policy data items in the format below:

    [
        {
          'key': <routing_policy_data_key1>,
          'rrdatas': <IP address list>,
          'forwarding_configs': <List of configs to be health checked>
        },
        {
          'key': <routing_policy_data_key2>,
          'rrdatas': <IP address list>,
          'forwarding_configs': <List of configs to be health checked>
        },
        ...
    ]

    Where <routing_policy_data_key> is either a weight or location name,
    depending on whether the user specified --routing-policy-type == WRR or
    --routing-policy-type == GEO, respectively. We keep
    <routing_policy_data_key> a string value, even in the case of weights
    (which will eventually be interpereted as floats). This is to keep this
    flag type generic between WRR and GEO types.
    """
        routing_policy_data = []
        policy_items = routing_policy_data_value.split(';')
        for policy_item in policy_items:
            key_value_split = policy_item.split('=')
            if len(key_value_split) != 2:
                raise arg_parsers.ArgumentTypeError('Must specify exactly one "=" inside each policy data item')
            key = key_value_split[0]
            value = key_value_split[1]
            ips = []
            forwarding_configs = []
            for val in value.split(','):
                if len(val.split('@')) == 2:
                    forwarding_configs.append(val)
                elif len(val.split('@')) == 1 and IsIPv4(val):
                    ips.append(val)
                elif len(val.split('@')) == 1:
                    forwarding_configs.append(val)
                else:
                    raise arg_parsers.ArgumentTypeError('Each policy rdata item should either be an ip address or a forwarding rule name optionally followed by its scope.')
            routing_policy_data.append({'key': key, 'rrdatas': ips, 'forwarding_configs': forwarding_configs})
        return routing_policy_data
    flag_name = '--routing_policy_data' if deprecated_name else '--routing-policy-data'
    return base.Argument(flag_name, metavar='ROUTING_POLICY_DATA', required=required, type=RoutingPolicyDataArgType, help='The routing policy data supports one of two formats below, depending on the choice of routing-policy-type.\n\nFor --routing-policy-type = "WRR" this flag indicates the weighted round robin policy data. The field accepts a semicolon-delimited list of the format "${weight_percent}=${rrdata},${rrdata}". Specify weight as a non-negative number (0 is allowed). Ratio of traffic routed to the target is calculated from the ratio of individual weight over the total across all weights.\n\nFor --routing-policy-type = "GEO" this flag indicates the geo-locations policy data. The field accepts a semicolon-delimited list of the format "${region}=${rrdata},${rrdata}". Each rrdata can either be an IP address or a reference to a forwarding rule of the format "FORWARDING_RULE_NAME", "FORWARDING_RULE_NAME@{region}", "FORWARDING_RULE_NAME@global", or the full resource path of the forwarding rule.')