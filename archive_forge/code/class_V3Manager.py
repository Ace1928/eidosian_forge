from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class V3Manager(BaseManager):

    def __init__(self, *args, **kwargs):
        super(V3Manager, self).__init__(**kwargs)
        self.required_resources = ['version', 'community', 'destination', 'port', 'network', 'security_name', 'auth_protocol', 'auth_password', 'security_level', 'privacy_protocol', 'privacy_password']
        self.want = V3Parameters(params=self.module.params)
        self.changes = V3Parameters()

    def _set_changed_options(self):
        changed = {}
        for key in V3Parameters.returnables:
            if getattr(self.want, key) is not None:
                changed[key] = getattr(self.want, key)
        if changed:
            self.changes = V3Parameters(params=changed)

    def _update_changed_options(self):
        changed = {}
        for key in V3Parameters.updatables:
            if getattr(self.want, key) is not None:
                attr1 = getattr(self.want, key)
                attr2 = getattr(self.have, key)
                if attr1 != attr2:
                    changed[key] = attr1
        if changed:
            self.changes = V3Parameters(params=changed)
            return True
        return False

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/sys/snmp/traps/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        return V3Parameters(params=response)