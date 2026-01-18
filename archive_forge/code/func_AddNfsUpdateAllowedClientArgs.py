from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNfsUpdateAllowedClientArgs(parser, hidden):
    """Adds NFS update allowed clients arguments group."""
    group_arg = parser.add_mutually_exclusive_group(required=False, hidden=hidden)
    group_arg.add_argument('--add-allowed-client', type=arg_parsers.ArgDict(spec=NFS_ALLOWED_CLIENT_SPEC, required_keys=REQUIRED_NFS_ALLOWED_CLIENT_KEYS), action='append', metavar='PROPERTY=VALUE', help=NFS_ALLOWED_CLIENTS_HELP_TEXT)
    group_arg.add_argument('--remove-allowed-client', type=arg_parsers.ArgDict(spec=REMOVE_NFS_ALLOWED_CLIENT_SPEC, required_keys=REQUIRED_REMOVE_NFS_ALLOWED_CLIENT_KEYS), action='append', metavar='PROPERTY=VALUE', help='\n              Removes an allowed client for the NFS share given its network name and cidr. This flag can be repeated to remove multiple allowed clients.\n\n              *network*::: The name of the network of the allowed client to remove.\n\n              *network-project-id*::: The project ID of the allowed client network.\n              If not present, the project ID of the NFS share will be used.\n\n              *cidr*::: The subnet of permitted IP addresses of the allowed client to remove.\n            ')
    group_arg.add_argument('--clear-allowed-clients', action='store_true', help='Removes all IP range reservations in the network.')