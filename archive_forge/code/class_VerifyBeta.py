from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp.kms_configs import client as kmsconfigs_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class VerifyBeta(Verify):
    """Verify that the Cloud NetApp Volumes KMS Config is reachable."""
    _RELEASE_TRACK = base.ReleaseTrack.BETA