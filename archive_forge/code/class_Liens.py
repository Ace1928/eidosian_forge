from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import liens
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Liens(base.Group):
    """Manage Cloud Resource Manager Liens.

  Commands to manage your Cloud Resource Liens.
  """

    @staticmethod
    def Args(parser):
        parser.display_info.AddUriFunc(liens.GetUri)