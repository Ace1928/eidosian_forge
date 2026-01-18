import copy
import logging
from openstack import exceptions as sdk_exceptions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _xform_volume_availability_zone(az):
    result = []
    zone_info = {'zone_name': az.name, 'zone_status': 'available' if az.state['available'] else 'not available'}
    result.append(zone_info)
    return result