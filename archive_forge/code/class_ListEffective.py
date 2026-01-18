from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc.manage.etd import clients
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc.manage import flags
from googlecloudsdk.command_lib.scc.manage import parsing
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA)
class ListEffective(base.ListCommand):
    """List the details of an Event Threat Detection effective custom module.

  List the details of resident and inherited Event Threat Detection custom
  modules for the specified folder or project with their effective enablement
  states. For an organization, this command lists only the custom modules
  that are created at the organization level. Custom modules created in child
  folders or projects are not included in the list.

  ## EXAMPLES

  To list resident and inherited Event Threat Detection custom modules
  with effective enablement states for organization 123, run:

  $ {command} --organization=organizations/123

  To list resident and inherited effective Event Threat Detection custom
  modules with effective enablement states for folder 456, run:

  $ {command} --folder=folders/456

  To list resident and inherited effective Event Threat Detection custom
  modules with effective enablement states for project 789, run:

  $ {command} --project=projects/789
  """

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        flags.CreateParentFlag(required=True).AddToParser(parser)

    def Run(self, args):
        parent = parsing.GetParentResourceNameFromArgs(args)
        page_size = args.page_size
        limit = args.limit
        client = clients.EffectiveETDCustomModuleClient()
        return client.List(page_size=page_size, parent=parent, limit=limit)