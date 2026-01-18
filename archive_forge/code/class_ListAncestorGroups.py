from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import properties
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ListAncestorGroups(base.ListCommand):
    """List ancestor groups of a specific service.

  List ancestor groups of a specific service.

  ## EXAMPLES

    List ancestor groups of service my-service:

   $ {command} my-service

   List ancestor groups of service my-service for a specific project '12345678':

    $ {command} my-service --project=12345678
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('service', help='Name of the service.')
        common_flags.add_resource_args(parser)
        base.PAGE_SIZE_FLAG.SetDefault(parser, 50)
        base.URI_FLAG.RemoveFromParser(parser)
        parser.display_info.AddFormat("\n          table(\n            groupName:label=''\n          )\n        ")

    def Run(self, args):
        """Run command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Resource name and its parent name.
    """
        if args.IsSpecified('folder'):
            resource_name = _FOLDER_RESOURCE_TEMPLATE % args.folder
        elif args.IsSpecified('organization'):
            resource_name = _ORGANIZATION_RESOURCE_TEMPLATE % args.organization
        elif args.IsSpecified('project'):
            resource_name = _PROJECT_RESOURCE_TEMPLATE % args.project
        else:
            project = properties.VALUES.core.project.Get(required=True)
            resource_name = _PROJECT_RESOURCE_TEMPLATE % project
        response = serviceusage.ListAncestorGroups(resource_name, _SERVICE_RESOURCE_TEMPLATE % args.service)
        return response