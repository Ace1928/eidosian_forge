import copy
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from oslo_log import log as logging
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as fwaas_const
def _extend_list(self, data, parsed_args):
    ext_data = copy.deepcopy(data)
    for d in ext_data:
        e_prefix = 'Event: '
        if d['event']:
            event = e_prefix + d['event'].upper()
        port = '(port) ' + d['target_id'] if d['target_id'] else ''
        resource_type = d['resource_type']
        if d['resource_id']:
            res = '(%s) %s' % (resource_type, d['resource_id'])
        else:
            res = ''
        t_prefix = 'Logged: '
        if port and res:
            t = '%s on %s' % (res, port)
        else:
            t = res + port
        target = t_prefix + t if t else t_prefix + '(None specified)'
        d['summary'] = ',\n'.join([event, target])
    return ext_data