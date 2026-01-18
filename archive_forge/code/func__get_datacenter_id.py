from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _get_datacenter_id(datacenters, identity):
    """
    Fetch and return datacenter UUID by datacenter name if found.
    """
    for datacenter in datacenters['items']:
        if identity in (datacenter['properties']['name'], datacenter['id']):
            return datacenter['id']
    return None