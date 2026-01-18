from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class TacacsParameters(BaseParameters):
    api_map = {'protocol': 'protocol_name', 'service': 'service_name'}
    api_attributes = ['authentication', 'accounting', 'protocol', 'service', 'secret', 'servers']
    returnables = ['servers', 'secret', 'authentication', 'accounting', 'service_name', 'protocol_name']
    updatables = ['servers', 'secret', 'authentication', 'accounting', 'service_name', 'protocol_name', 'auth_source']