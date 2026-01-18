from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
def ExtractZonesFromLocations(response, _):
    """Returns the zones from a ListLocationResponse."""
    for region in response:
        if not region.metadata:
            continue
        metadata = encoding.MessageToDict(region.metadata)
        for zone in metadata.get('availableZones', []):
            yield _Zone(name=zone, region=region.locationId)