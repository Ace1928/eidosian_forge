from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.core import exceptions
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class RemoveBgpPeerBeta(RemoveBgpPeer):
    """Remove a BGP peer from a Compute Engine router."""
    ROUTER_ARG = None

    @classmethod
    def Args(cls, parser):
        cls._Args(parser)

    def Run(self, args):
        return self._Run(args)