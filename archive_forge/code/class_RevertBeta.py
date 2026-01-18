from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp.volumes import client as volumes_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp.volumes import flags as volumes_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class RevertBeta(Revert):
    """Revert a Cloud NetApp Volume back to a specified Snapshot."""
    _RELEASE_TRACK = base.ReleaseTrack.BETA