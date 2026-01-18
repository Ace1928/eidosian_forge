from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip_network
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _format_options(self, options):
    negate = None
    cleaned = dict(((k, v) for k, v in iteritems(options) if v is not None))
    if 'country' in cleaned.keys() and 'state' in cleaned.keys():
        del cleaned['country']
    if 'negate' in cleaned.keys():
        negate = cleaned['negate']
        del cleaned['negate']
    name, value = cleaned.popitem()
    if negate:
        result = '{0} {1} {2}'.format(negate, name, value)
        return result
    result = '{0} {1}'.format(name, value)
    return result