from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddBGPPeerConfigs(bgp_node_pool_config_group, is_update=False):
    """Adds a flag for BGP peer config field.

  Args:
    bgp_node_pool_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    bgp_peer_config_help_text = "\nList of BGP peers that the cluster will connect to. At least one peer must be configured for each control plane node.\n\nExamples:\n\nTo specify configurations for two peers of BGP autonomous system number (ASN) 10000 and 20000,\n\n```\n$ {command} example_cluster\n      --bgp-peer-configs 'asn=10000,ip=192.168.1.1,control-plane-nodes=192.168.1.2;192.168.1.3'\n      --bgp-peer-configs 'asn=20000,ip=192.168.2.1,control-plane-nodes=192.168.2.2;192.168.2.3'\n```\n\nUse quote around the flag value to escape semicolon in the terminal.\n\n  "
    required = not is_update
    bgp_node_pool_config_group.add_argument('--bgp-peer-configs', help=bgp_peer_config_help_text, action='append', required=required, type=arg_parsers.ArgDict(spec={'asn': int, 'ip': str, 'control-plane-nodes': arg_parsers.ArgList(custom_delim_char=';')}, required_keys=['asn', 'ip']), metavar='asn=ASN,ip=IP,control-plane-nodes=NODE_IP_1;NODE_IP_2')