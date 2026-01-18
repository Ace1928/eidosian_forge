from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.billing import billing_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.billing import flags
from googlecloudsdk.command_lib.billing import utils
class Unlink(base.Command):
    """Unlink the account (if any) linked with a project."""
    detailed_help = {'DESCRIPTION': '          This command unlinks a project from its linked billing\n          account. This disables billing on the project.\n          ', 'EXAMPLES': '          To unlink the project `my-project` from its linked billing account,\n          run:\n\n            $ {command} my-project\n          '}

    @staticmethod
    def Args(parser):
        flags.GetProjectIdArgument().AddToParser(parser)

    def Run(self, args):
        client = billing_client.ProjectsClient()
        project_ref = utils.ParseProject(args.project_id)
        return client.Link(project_ref, None)