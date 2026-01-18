from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kmsinventory import inventory
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import resource_args
from googlecloudsdk.command_lib.resource_manager import completers
class SearchProtectedResources(base.ListCommand):
    """Searches the resources protected by a key."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddKmsKeyResourceArgForKMS(parser, True, '--keyname')
        parser.add_argument('--scope', metavar='ORGANIZATION_ID', completer=completers.OrganizationCompleter, required=True, help='Organization ID.')
        parser.add_argument('--resource-types', metavar='RESOURCE_TYPES', type=arg_parsers.ArgList(), help=RESOURCE_TYPE_HELP)

    def Run(self, args):
        key_name = args.keyname
        org = args.scope
        resource_types = args.resource_types
        if not resource_types:
            resource_types = []
        return inventory.SearchProtectedResources(scope=org, key_name=key_name, resource_types=resource_types, args=args)