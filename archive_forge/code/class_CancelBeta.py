from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.filestore.instances import flags as instances_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class CancelBeta(Cancel):
    """Cancel a Filestore operation."""
    _API_VERSION = filestore_client.BETA_API_VERSION