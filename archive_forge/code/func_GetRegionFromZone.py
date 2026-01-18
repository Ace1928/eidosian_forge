from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetRegionFromZone(zone):
    """Returns the GCP region that the input zone is in."""
    return '-'.join(zone.split('-')[:-1])