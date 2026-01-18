from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.projects import util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class NodePools(base.Group):
    """Create and manage node pools in an Anthos cluster on VMware."""

    @staticmethod
    def Args(parser: parser_arguments.ArgumentInterceptor):
        pass