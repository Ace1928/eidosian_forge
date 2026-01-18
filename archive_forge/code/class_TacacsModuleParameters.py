from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class TacacsModuleParameters(TacacsParameters):

    @property
    def servers(self):
        if self._values['servers'] is None:
            return None
        result = []
        for server in self._values['servers']:
            if isinstance(server, dict):
                if 'address' not in server:
                    raise F5ModuleError("An 'address' field must be provided when specifying separate fields to the 'servers' parameter.")
                address = server.get('address')
                port = server.get('port', None)
            elif isinstance(server, string_types):
                address = server
                port = None
            if port is None:
                result.append('{0}'.format(address))
            else:
                result.append('{0}:{1}'.format(address, port))
        return result

    @property
    def auth_source(self):
        return 'tacacs'