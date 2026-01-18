import copy
import logging
from openstack import exceptions as sdk_exceptions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _xform_compute_availability_zone(az, include_extra):
    result = []
    zone_info = {'zone_name': az.name, 'zone_status': 'available' if az.state['available'] else 'not available'}
    if not include_extra:
        result.append(zone_info)
        return result
    if az.hosts:
        for host, services in az.hosts.items():
            host_info = copy.deepcopy(zone_info)
            host_info['host_name'] = host
            for svc, state in services.items():
                info = copy.deepcopy(host_info)
                info['service_name'] = svc
                info['service_status'] = '%s %s %s' % ('enabled' if state['active'] else 'disabled', ':-)' if state['available'] else 'XXX', state['updated_at'])
                result.append(info)
    else:
        zone_info['host_name'] = ''
        zone_info['service_name'] = ''
        zone_info['service_status'] = ''
        result.append(zone_info)
    return result