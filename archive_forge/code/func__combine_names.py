from __future__ import absolute_import, division, print_function
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _combine_names(self, item):
    server_name = transform_name(item['partition'], item['server_name'])
    virtual_server = transform_name(name=item['virtual_server'])
    result = '{0}:{1}'.format(server_name, virtual_server)
    return result