import copy
import logging
from openstack import exceptions as sdk_exceptions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _xform_network_availability_zone(az):
    result = []
    zone_info = {}
    zone_info['zone_name'] = az.name
    zone_info['zone_status'] = az.state
    if 'unavailable' == zone_info['zone_status']:
        zone_info['zone_status'] = 'not available'
    zone_info['zone_resource'] = az.resource
    result.append(zone_info)
    return result