from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Microservices(base.Group):
    """Manage microservices functionalities.

  {command} group lets you manage functionalities for microservices on
  the Google Cloud Platform.
  """
    category = base.MICROSERVICES_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args