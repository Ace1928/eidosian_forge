from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateAccessConfigInstancesBeta(UpdateAccessConfigInstances):
    """Update a Compute Engine virtual machine access configuration."""
    _support_public_dns = False
    _support_network_tier = False