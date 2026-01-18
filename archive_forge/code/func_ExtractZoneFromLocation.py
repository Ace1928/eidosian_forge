from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
def ExtractZoneFromLocation(response, args):
    """Returns the argument-specified zone from a GetLocationResponse."""
    metadata = encoding.MessageToDict(response.metadata)
    want_zone = args.zone.split('/')[-1]
    for zone_name, zone in metadata.get('availableZones', {}).items():
        if zone_name == want_zone:
            if 'rackTypes' in zone:
                racks = zone.pop('rackTypes')
                populated_rack = []
                for rack, rack_type in racks.items():
                    if rack_type == 'BASE':
                        populated_rack.append(rack + ' (BASE)')
                    elif rack_type == 'EXPANSION':
                        populated_rack.append(rack + ' (EXPANSION)')
                    else:
                        populated_rack.append(rack)
                zone['racks'] = populated_rack
            return zone
    raise exceptions.Error('Zone not found: {}'.format(want_zone))