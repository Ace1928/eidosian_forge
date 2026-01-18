from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_chain_value(self, item, true_name):
    if 'chain' not in item or item['chain'] in ('none', None, 'None'):
        result = 'none'
    else:
        result = self._cert_filename(fq_name(self.partition, item['chain']), true_name)
    return result