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
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class MutateDeployedIndexV1Beta1(MutateDeployedIndexV1):
    """Mutate an existing deployed index from a Vertex AI index endpoint."""
    detailed_help = DETAILED_HELP

    def Run(self, args):
        return self._Run(args, constants.BETA_VERSION)