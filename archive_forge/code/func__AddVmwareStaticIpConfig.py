from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddVmwareStaticIpConfig(ip_configuration_mutex_group):
    """Adds flags to specify Static IP configuration.

  Args:
    ip_configuration_mutex_group: The parent group to add the flag to.
  """
    static_ip_config_from_file_help_text = '\nPath of the YAML/JSON file that contains the static IP configurations, used by Anthos on VMware user cluster node pools.\n\nExamples:\n\n    staticIPConfig:\n      ipBlocks:\n      - gateway: 10.251.31.254\n        netmask: 255.255.252.0\n        ips:\n        - hostname: hostname-1\n          ip: 1.1.1.1\n        - hostname: hostname-2\n          ip: 2.2.2.2\n        - hostname: hostname-3\n          ip: 3.3.3.3\n        - hostname: hostname-4\n          ip: 4.4.4.4\n\nList of supported fields in `staticIPConfig`\n\nKEY       | VALUE                 | NOTE\n--------- | --------------------  | -----------------\nipBlocks  | one or more ipBlocks  | required, mutable\n\nList of supported fields in `ipBlocks`\n\nKEY     | VALUE           | NOTE\n------- | --------------- | -------------------\ngateway | IP address      | required, immutable\nnetmask | IP address      | required, immutable\nips     | one or more ips | required, mutable\n\nList of supported fields in `ips`\n\nKEY       | VALUE       | NOTE\n--------- | ----------- | -------------------\nhostname  | string      | optional, immutable\nip        | IP address  | required, immutable\n\nNew `ips` fields can be added, existing `ips` fields cannot be removed or updated.\n'
    static_ip_config_mutex_group = ip_configuration_mutex_group.add_group(help='Static IP configuration group', mutex=True)
    static_ip_config_mutex_group.add_argument('--static-ip-config-from-file', help=static_ip_config_from_file_help_text, type=arg_parsers.YAMLFileContents(), hidden=True)
    static_ip_config_ip_blocks_help_text = "\nStatic IP configurations.\n\nExpect an individual IP address, an individual IP address with an optional hostname, or a CIDR block.\n\nExample:\n\nTo specify two Static IP blocks,\n\n```\n$ gcloud {command}\n    --static-ip-config-ip-blocks 'gateway=192.168.0.1,netmask=255.255.255.0,ips=192.168.1.1;0.0.0.0 localhost;192.168.1.2/16'\n    --static-ip-config-ip-blocks 'gateway=192.168.1.1,netmask=255.255.0.0,ips=8.8.8.8;4.4.4.4'\n```\n\nUse quote around the flag value to escape semicolon in the terminal.\n  "
    static_ip_config_mutex_group.add_argument('--static-ip-config-ip-blocks', help=static_ip_config_ip_blocks_help_text, action='append', type=arg_parsers.ArgDict(spec={'gateway': str, 'netmask': str, 'ips': arg_parsers.ArgList(element_type=_ParseStaticIpConfigIpBlock, custom_delim_char=';')}))