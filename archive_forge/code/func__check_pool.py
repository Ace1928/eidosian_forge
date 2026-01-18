from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _check_pool(self, item):
    pool = transform_name(name=fq_name(self.partition, item))
    uri = 'https://{0}:{1}/mgmt/tm/ltm/pool/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], pool)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError:
        return False
    if resp.status == 404 or ('code' in response and response['code'] == 404):
        raise F5ModuleError('The specified pool {0} does not exist.'.format(pool))
    return item