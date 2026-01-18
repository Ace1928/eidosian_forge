from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration.conversion_workspaces import flags as cw_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DescribeIssues(base.Command):
    """Describe issues in a Database Migration Service conversion workspace."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        resource_args.AddConversionWorkspaceResourceArg(parser, 'to describe issues')
        cw_flags.AddCommitIdFlag(parser)
        cw_flags.AddUncomittedFlag(parser)
        cw_flags.AddFilterFlag(parser)
        parser.display_info.AddFormat('\n          table(\n            parentEntity:label=PARENT,\n            shortName:label=NAME,\n            entityType:label=ENTITY_TYPE,\n            issueType:label=ISSUE_TYPE,\n            issueSeverity:label=ISSUE_SEVERITY,\n            issueCode:label=ISSUE_CODE,\n            issueMessage:label=ISSUE_MESSAGE\n          )\n        ')

    def Run(self, args):
        """Describe the database entity issues for a DMS conversion workspace.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A list of database entity issues for the specified conversion workspace
      and arguments.
    """
        conversion_workspace_ref = args.CONCEPTS.conversion_workspace.Parse()
        cw_client = conversion_workspaces.ConversionWorkspacesClient(self.ReleaseTrack())
        return cw_client.DescribeIssues(conversion_workspace_ref.RelativeName(), args)