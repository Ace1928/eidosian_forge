from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration import flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import alloydb_flags as ad_flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import create_helper
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AlloyDB(base.Command):
    """Create a Database Migration Service connection profile for AlloyDB."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        resource_args.AddAlloyDBConnectionProfileResourceArgs(parser, 'to create')
        cp_flags.AddNoAsyncFlag(parser)
        cp_flags.AddDisplayNameFlag(parser)
        ad_flags.AddPasswordFlag(parser)
        ad_flags.AddNetworkFlag(parser)
        ad_flags.AddClusterLabelsFlag(parser)
        ad_flags.AddPrimaryIdFlag(parser)
        ad_flags.AddCpuCountFlag(parser)
        ad_flags.AddDatabaseFlagsFlag(parser)
        ad_flags.AddPrimaryLabelsFlag(parser)
        ad_flags.AddDatabaseVersionFlag(parser)
        flags.AddLabelsCreateFlags(parser)

    def Run(self, args):
        """Create a Database Migration Service connection profile for AlloyDB.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A dict object representing the operations resource describing the create
      operation if the create was successful.
    """
        connection_profile_ref = args.CONCEPTS.connection_profile.Parse()
        parent_ref = connection_profile_ref.Parent().RelativeName()
        helper = create_helper.CreateHelper()
        return helper.create(self.ReleaseTrack(), parent_ref, connection_profile_ref, args, 'ALLOYDB')