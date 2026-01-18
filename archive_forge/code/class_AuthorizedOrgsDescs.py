from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class AuthorizedOrgsDescs(base.Group):
    """Manage Access Context Manager authorized organizations descriptions.

   An authorized organizations description describes a list of organizations (1)
   that have been authorized to use certain asset (for example, device) data
   owned by different organizations at the enforcement points, or (2) with
   certain asset (for example, device) have been authorized to access the
   resources in another organization at the enforcement points.
  """