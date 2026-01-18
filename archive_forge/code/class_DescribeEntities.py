from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration.conversion_workspaces import flags as cw_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DescribeEntities(base.Command):
    """Describe database entities in a Database Migration conversion workspace."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddConversionWorkspaceResourceArg(parser, 'describe entities')
        cw_flags.AddTreeTypeFlag(parser, required=True)
        cw_flags.AddCommitIdFlag(parser)
        cw_flags.AddUncomittedFlag(parser)
        cw_flags.AddFilterFlag(parser)
        parser.display_info.AddFormat('\n          table(\n            parentEntity:label=PARENT,\n            shortName:label=NAME,\n            tree:label=TREE_TYPE,\n            entityType:label=ENTITY_TYPE\n          )\n        ')

    def Run(self, args):
        """Describe database entities for a DMS conversion workspace.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A list of entities for the specified conversion workspace and arguments.
    """
        conversion_workspace_ref = args.CONCEPTS.conversion_workspace.Parse()
        cw_client = conversion_workspaces.ConversionWorkspacesClient(self.ReleaseTrack())
        return cw_client.DescribeEntities(conversion_workspace_ref.RelativeName(), args)