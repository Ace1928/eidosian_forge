from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import properties
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ListGroupMembers(base.ListCommand):
    """List members of a specific service and group.

  List members of a specific service and group.

  ## EXAMPLES

   List members of service my-service and group my-group:

   $ {command} my-service my-group

   List members of service my-service and group my-group
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
        parser.display_info.AddFormat("\n          table(\n            name:label=''\n          )\n        ")

    def Run(self, args):
        """Run command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Resource name and its parent name.
    """
        if args.IsSpecified('folder'):
            resource_name = _FOLDER_RESOURCE.format(args.folder)
        elif args.IsSpecified('organization'):
            resource_name = _ORGANIZATION_RESOURCE.format(args.organization)
        elif args.IsSpecified('project'):
            resource_name = _PROJECT_RESOURCE.format(args.project)
        else:
            project = properties.VALUES.core.project.Get(required=True)
            resource_name = _PROJECT_RESOURCE.format(project)
        response = serviceusage.ListGroupMembersV2Alpha(resource_name, '{}/{}'.format(_SERVICE_RESOURCE.format(args.service), _GROUP_RESOURCE.format(args.group)), args.page_size)
        group_members = []
        result = collections.namedtuple('ListMembers', ['name'])
        for member_list in response:
            member = member_list.member
            if member.groupName:
                group_members.append(result(member.groupName))
            else:
                group_members.append(result(member.serviceName))
        return group_members