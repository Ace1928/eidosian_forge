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
def GetResourceRecordSetsRoutingPolicyBackupDataArg(required=False):
    """Returns --routing-policy-backup-data command line arg value."""

    def RoutingPolicyBackupDataArg(routing_policy_backup_data):
        """Converts --routing-policy-backup-data flag value to a list of policy data items.

    Args:
      routing_policy_backup_data: String value specified in the
        --routing-policy-backup-data flag.

    Returns:
      A list of policy data items in the format below:

    [
        {
          'key': <location1>,
          'rrdatas': <IP address list>,
          'forwarding_configs': <List of configs to be health checked>
        },
        {
          'key': <location2>,
          'rrdatas': <IP address list>,
          'forwarding_configs': <List of configs to be health checked>
        },
        ...
    ]
    """
        backup_data = []
        policy_items = routing_policy_backup_data.split(';')
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
            backup_data.append({'key': key, 'rrdatas': ips, 'forwarding_configs': forwarding_configs})
        return backup_data
    return base.Argument('--routing-policy-backup-data', metavar='ROUTING_POLICY_BACKUP_DATA', required=required, type=RoutingPolicyBackupDataArg, help='The backup configuration for a primary backup routing policy. This configuration has the same format as the routing-policy-data argument because it is just another geo-locations policy.')