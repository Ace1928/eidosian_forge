import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _treeizeAvailabilityZone(zone):
    """Build a tree view for availability zones."""
    AvailabilityZone = availability_zones.AvailabilityZone
    az = AvailabilityZone(zone.manager, zone.to_dict(), zone._loaded)
    result = []
    az.zoneName = zone.zoneName
    az.zoneState = 'available' if zone.zoneState['available'] else 'not available'
    az.set_info('zoneName', az.zoneName)
    az.set_info('zoneState', az.zoneState)
    result.append(az)
    if zone.hosts is not None:
        zone_hosts = sorted(zone.hosts.items(), key=lambda x: x[0])
        for host, services in zone_hosts:
            az = AvailabilityZone(zone.manager, zone.to_dict(), zone._loaded)
            az.zoneName = '|- %s' % host
            az.zoneState = ''
            az.set_info('zoneName', az.zoneName)
            az.set_info('zoneState', az.zoneState)
            result.append(az)
            for svc, state in services.items():
                az = AvailabilityZone(zone.manager, zone.to_dict(), zone._loaded)
                az.zoneName = '| |- %s' % svc
                az.zoneState = '%s %s %s' % ('enabled' if state['active'] else 'disabled', ':-)' if state['available'] else 'XXX', state['updated_at'])
                az.set_info('zoneName', az.zoneName)
                az.set_info('zoneState', az.zoneState)
                result.append(az)
    return result