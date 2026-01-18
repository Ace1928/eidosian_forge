from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ExternalAccountKeys(base.Group):
    """Create ACME external account binding keys.

  {command} lets you create an external account key associated with
  Google Trust Services' publicly trusted certificate authority.

  The external account key will be associated with the Cloud project and
  it may be bound to an Automatic Certificate Management Environment (ACME)
  account following RFC 8555.

  See https://tools.ietf.org/html/rfc8555#section-7.3.4 for more details.
  """