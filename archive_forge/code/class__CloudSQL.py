from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration import flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import cloudsql_flags as cs_flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import create_helper
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
class _CloudSQL(object):
    """Create a Database Migration Service connection profile for Cloud SQL."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        resource_args.AddCloudSqlConnectionProfileResourceArgs(parser, 'to create')
        cp_flags.AddNoAsyncFlag(parser)
        cp_flags.AddDisplayNameFlag(parser)
        cp_flags.AddProviderFlag(parser)
        cs_flags.AddActivationPolicylag(parser)
        cs_flags.AddAuthorizedNetworksFlag(parser)
        cs_flags.AddAutoStorageIncreaseFlag(parser)
        cs_flags.AddDatabaseFlagsFlag(parser)
        cs_flags.AddDataDiskSizeFlag(parser)
        cs_flags.AddDataDiskTypeFlag(parser)
        cs_flags.AddAvailabilityTypeFlag(parser)
        cs_flags.AddEnableIpv4Flag(parser)
        cs_flags.AddPrivateNetworkFlag(parser)
        cs_flags.AddRequireSslFlag(parser)
        cs_flags.AddUserLabelsFlag(parser)
        cs_flags.AddStorageAutoResizeLimitFlag(parser)
        cs_flags.AddTierFlag(parser)
        cs_flags.AddZoneFlag(parser)
        cs_flags.AddSecondaryZoneFlag(parser)
        cs_flags.AddRootPassword(parser)
        flags.AddLabelsCreateFlags(parser)

    def Run(self, args):
        """Create a Database Migration Service connection profile for Cloud SQL.

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
        return helper.create(self.ReleaseTrack(), parent_ref, connection_profile_ref, args, 'CLOUDSQL')