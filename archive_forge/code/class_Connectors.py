from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Connectors(base.Group):
    """Create and manipulate beyondcorp connectors.

  A Beyondcorp connector represents an application facing component deployed
  proximal to and with direct access to the application instances. It is used to
  establish connectivity between the remote enterprise environment and Google
  Cloud Platform. It initiates connections to the applications and can proxy the
  data from users over the connection.
  """