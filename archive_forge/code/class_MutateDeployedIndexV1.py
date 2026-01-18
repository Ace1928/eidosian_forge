from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.index_endpoints import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import index_endpoints_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class MutateDeployedIndexV1(base.Command):
    """Mutate an existing deployed index from a Vertex AI index endpoint."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddIndexEndpointResourceArg(parser, 'ID of the index endpoint.')
        flags.GetDeployedIndexId().AddToParser(parser)
        flags.AddMutateDeploymentResourcesArgs(parser, 'deployed index')
        flags.AddReservedIpRangesArgs(parser, 'deployed index')
        flags.AddDeploymentGroupArg(parser)
        flags.AddAuthConfigArgs(parser, 'deployed index')
        flags.GetEnableAccessLoggingArg().AddToParser(parser)

    def _Run(self, args, version):
        index_endpoint_ref = args.CONCEPTS.index_endpoint.Parse()
        region = index_endpoint_ref.AsDict()['locationsId']
        with endpoint_util.AiplatformEndpointOverrides(version, region=region):
            index_endpoint_client = client.IndexEndpointsClient(version=version)
            if version == constants.GA_VERSION:
                operation = index_endpoint_client.MutateDeployedIndex(index_endpoint_ref, args)
            else:
                operation = index_endpoint_client.MutateDeployedIndexBeta(index_endpoint_ref, args)
            response_msg = operations_util.WaitForOpMaybe(operations_client=operations.OperationsClient(version=version), op=operation, op_ref=index_endpoints_util.ParseIndexEndpointOperation(operation.name))
        if response_msg is not None:
            response = encoding.MessageToPyValue(response_msg)
            if 'deployedIndex' in response and 'id' in response['deployedIndex']:
                log.status.Print('Id of the deployed index updated: {}.'.format(response['deployedIndex']['id']))
        return response_msg

    def Run(self, args):
        return self._Run(args, constants.GA_VERSION)