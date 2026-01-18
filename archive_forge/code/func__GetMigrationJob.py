from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _GetMigrationJob(self, source_ref, destination_ref, conversion_workspace_ref, cmek_key_ref, args):
    """Returns a migration job."""
    migration_job_type = self.messages.MigrationJob
    labels = labels_util.ParseCreateArgs(args, self.messages.MigrationJob.LabelsValue)
    type_value = self._GetType(migration_job_type, args.type)
    source = source_ref.RelativeName()
    destination = destination_ref.RelativeName()
    params = {}
    if args.IsSpecified('peer_vpc'):
        params['vpcPeeringConnectivity'] = self._GetVpcPeeringConnectivity(args)
    elif args.IsSpecified('vm_ip'):
        params['reverseSshConnectivity'] = self._GetReverseSshConnectivity(args)
    elif args.IsSpecified('static_ip'):
        params['staticIpConnectivity'] = self._GetStaticIpConnectivity()
    migration_job_obj = migration_job_type(labels=labels, displayName=args.display_name, state=migration_job_type.StateValueValuesEnum.CREATING, type=type_value, dumpPath=args.dump_path, source=source, destination=destination, **params)
    if conversion_workspace_ref is not None:
        migration_job_obj.conversionWorkspace = self._GetConversionWorkspaceInfo(conversion_workspace_ref, args)
    if cmek_key_ref is not None:
        migration_job_obj.cmekKeyName = cmek_key_ref.RelativeName()
    if args.IsKnownAndSpecified('filter'):
        args.filter, server_filter = filter_rewrite.Rewriter().Rewrite(args.filter)
        migration_job_obj.filter = server_filter
    if args.IsKnownAndSpecified('dump_parallel_level'):
        migration_job_obj.performanceConfig = self._GetPerformanceConfig(args)
    if args.IsKnownAndSpecified('dump_type'):
        migration_job_obj.dumpType = self._GetDumpType(self.messages.MigrationJob, args.dump_type)
    if args.IsKnownAndSpecified('sqlserver_databases'):
        migration_job_obj.sqlserverHomogeneousMigrationJobConfig = self._GetSqlserverHomogeneousMigrationJobConfig(args)
    return migration_job_obj