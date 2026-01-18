from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration.conversion_workspaces import flags as cw_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DescribeDDLs(base.Command):
    """Describe DDLs in a Database Migration Service conversion workspace."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        resource_args.AddConversionWorkspaceResourceArg(parser, 'to describe DDLs')
        cw_flags.AddTreeTypeFlag(parser, required=False)
        cw_flags.AddCommitIdFlag(parser)
        cw_flags.AddUncomittedFlag(parser)
        cw_flags.AddFilterFlag(parser)
        parser.display_info.AddFormat('table(ddl:label=DDLs)')

    def Run(self, args):
        """Describe the DDLs for a Database Migration Service conversion workspace.

    Args:
      args: argparse.Namespace, the arguments that this command was invoked
        with.

    Returns:
      A list of DDLs for the specified conversion workspace and arguments.
    """
        conversion_workspace_ref = args.CONCEPTS.conversion_workspace.Parse()
        cw_client = conversion_workspaces.ConversionWorkspacesClient(self.ReleaseTrack())
        return cw_client.DescribeDDLs(conversion_workspace_ref.RelativeName(), args)