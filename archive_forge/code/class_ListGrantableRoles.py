from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.iam import exceptions
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import flags
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import resources
class ListGrantableRoles(base.Command):
    """List IAM grantable roles for a resource.

  This command displays the list of grantable roles for a resource.
  The resource can be referenced either via the full resource name or via a URI.
  User can then add IAM policy bindings to grant the roles.
  """
    detailed_help = {'EXAMPLES': textwrap.dedent('          List grantable roles for a project:\n\n            $ {command} //cloudresourcemanager.googleapis.com/projects/PROJECT_ID\n\n          List grantable roles for a resource identified via full resource name:\n\n            $ {command} //compute.googleapis.com/projects/example-project/zones/us-central1-f/instances/example-instance\n\n          List grantable roles for a resource identified via URI:\n\n            $ {command} \\\n            https://www.googleapis.com/compute/v1/projects/example-project/zones/us-central1-f/instances/example-instance\n      ')}

    @staticmethod
    def Args(parser):
        flags.GetResourceNameFlag('get the list of roles for').AddToParser(parser)
        base.FILTER_FLAG.AddToParser(parser)
        base.PAGE_SIZE_FLAG.AddToParser(parser)
        base.PAGE_SIZE_FLAG.SetDefault(parser, 100)

    def Run(self, args):
        resource = None
        if args.resource.startswith('//'):
            resource = args.resource
        if args.resource.startswith('http'):
            resource_ref = resources.REGISTRY.Parse(args.resource)
            resource = iam_util.GetFullResourceName(resource_ref)
        if not resource:
            raise exceptions.InvalidResourceException('The given resource is not a valid full resource name or URL.')
        client, messages = util.GetClientAndMessages()
        return list_pager.YieldFromList(client.roles, messages.QueryGrantableRolesRequest(fullResourceName=resource), field='roles', method='QueryGrantableRoles', batch_size=args.page_size, batch_size_attribute='pageSize')