from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import properties
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ListDescendantServices(base.ListCommand):
    """List descendant services of a specific service and group.

  List descendant services of a specific service and group.

  ## EXAMPLES

   List descendant services of service my-service and group my-group:

   $ {command} my-service my-group

   List descendant services of service my-service and group my-group
   for a specific project '12345678':

    $ {command} my-service my-group --project=12345678
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('service', help='Name of the service.')
        parser.add_argument('group', help='Service group name, for example "dependencies".')
        common_flags.add_resource_args(parser)
        base.PAGE_SIZE_FLAG.SetDefault(parser, 50)
        base.URI_FLAG.RemoveFromParser(parser)
        parser.display_info.AddFormat("\n          table(\n            serviceName:label=''\n          )\n        ")

    def Run(self, args):
        """Run command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Resource name and its parent name.
    """
        if args.IsSpecified('folder'):
            resource_name = _FOLDER_RESOURCE % args.folder
        elif args.IsSpecified('organization'):
            resource_name = _ORGANIZATION_RESOURCE % args.organization
        elif args.IsSpecified('project'):
            resource_name = _PROJECT_RESOURCE % args.project
        else:
            project = properties.VALUES.core.project.Get(required=True)
            resource_name = _PROJECT_RESOURCE % project
        response = serviceusage.ListDescendantServices(resource_name, '{}/{}'.format(_SERVICE_RESOURCE % args.service, _GROUP_RESOURCE % args.group))
        return response