from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkeonprem import operations
from googlecloudsdk.api_lib.container.gkeonprem import vmware_admin_clusters as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.container.gkeonprem import constants
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.container.vmware import constants as vmware_constants
from googlecloudsdk.command_lib.container.vmware import flags as vmware_flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Enroll(base.Command):
    """Enroll an Anthos on VMware admin cluster."""
    detailed_help = {'EXAMPLES': _EXAMPLES}

    @staticmethod
    def Args(parser: parser_arguments.ArgumentInterceptor):
        parser.display_info.AddFormat(vmware_constants.VMWARE_CLUSTERS_FORMAT)
        flags.AddAdminClusterMembershipResourceArg(parser, positional=False)
        vmware_flags.AddAdminClusterResourceArg(parser, 'to enroll')
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        cluster_client = apis.AdminClustersClient()
        admin_cluster_ref = args.CONCEPTS.admin_cluster.Parse()
        operation = cluster_client.Enroll(args)
        if args.async_ and (not args.IsSpecified('format')):
            args.format = constants.OPERATIONS_FORMAT
        if args.async_:
            operations.log_enroll(admin_cluster_ref, args.async_)
            return operation
        else:
            operation_client = operations.OperationsClient()
            operation_response = operation_client.Wait(operation)
            operations.log_enroll(admin_cluster_ref, args.async_)
            return operation_response