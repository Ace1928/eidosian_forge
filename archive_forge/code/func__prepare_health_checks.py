import sys
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from saharaclient.osc import utils
def _prepare_health_checks(data):
    additional_data = {}
    ver = data.get('verification', {})
    additional_fields = ['verification_status']
    additional_data['verification_status'] = ver.get('status', 'UNKNOWN')
    for check in ver.get('checks', []):
        row_name = 'Health check (%s)' % check['name']
        additional_data[row_name] = check['status']
        additional_fields.append(row_name)
    return (additional_data, additional_fields)