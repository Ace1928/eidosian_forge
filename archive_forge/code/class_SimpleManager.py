from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class SimpleManager(BaseManager):

    def __init__(self, *args, **kwargs):
        super(SimpleManager, self).__init__(**kwargs)
        self.want = SimpleParameters(params=self.module.params)
        self.have = SimpleParameters()
        self.changes = SimpleChanges()

    def _set_changed_options(self):
        changed = {}
        for key in SimpleParameters.returnables:
            if getattr(self.want, key) is not None:
                changed[key] = getattr(self.want, key)
        if changed:
            self.changes = SimpleChanges(params=changed)

    def _update_changed_options(self):
        diff = Difference(self.want, self.have)
        updatables = SimpleParameters.updatables
        changed = dict()
        for k in updatables:
            change = diff.compare(k)
            if change is None:
                continue
            else:
                changed[k] = change
        if changed:
            self.changes = SimpleChanges(params=changed)
            return True
        return False

    def exec_module(self):
        start = datetime.now().isoformat()
        version = tmos_version(self.client)
        changed = False
        result = dict()
        state = self.want.state
        if state == 'draft':
            raise F5ModuleError("The 'draft' status is not available on BIG-IP versions < 12.1.0")
        if state == 'present':
            changed = self.present()
        elif state == 'absent':
            changed = self.absent()
        changes = self.changes.to_return()
        result.update(**changes)
        result.update(dict(changed=changed))
        self._announce_deprecations()
        self._announce_warnings()
        send_teem(start, self.client, self.module, version)
        return result

    def create(self):
        self._validate_creation_parameters()
        self._set_changed_options()
        if self.module.check_mode:
            return True
        self.create_on_device()
        return True

    def update(self):
        self.have = self.read_current_from_device()
        if not self.should_update():
            return False
        if self.module.check_mode:
            return True
        self.update_on_device()
        return True

    def absent(self):
        changed = False
        if self.exists():
            changed = self.remove()
        return changed

    def remove(self):
        if self.module.check_mode:
            return True
        self.remove_from_device()
        if self.exists():
            raise F5ModuleError('Failed to delete the policy')
        return True

    def exists(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            return False
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        errors = [401, 403, 409, 500, 501, 502, 503, 504]
        if resp.status in errors or ('code' in response and response['code'] in errors):
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        query = '?expandSubcollections=true'
        resp = self.client.api.get(uri + query)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        rules = self._get_rule_names(response['rulesReference'])
        result = SimpleParameters(params=response)
        result.update(dict(rules=rules))
        return result

    def update_on_device(self):
        params = self.changes.api_params()
        if params:
            uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
            resp = self.client.api.patch(uri, json=params)
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if 'code' in response and response['code'] == 400:
                if 'message' in response:
                    raise F5ModuleError(response['message'])
                else:
                    raise F5ModuleError(resp.content)
        self._upsert_policy_rules_on_device()

    def create_on_device(self):
        params = self.want.api_params()
        payload = dict(name=self.want.name, partition=self.want.partition, **params)
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=payload)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 403]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        self._upsert_policy_rules_on_device()
        return True

    def remove_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        response = self.client.api.delete(uri)
        if response.status == 200:
            return True
        raise F5ModuleError(response.content)