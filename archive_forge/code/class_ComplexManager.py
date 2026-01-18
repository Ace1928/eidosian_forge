from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class ComplexManager(BaseManager):

    def __init__(self, *args, **kwargs):
        super(ComplexManager, self).__init__(**kwargs)
        self.want = ComplexParameters(params=self.module.params)
        self.have = ComplexParameters()
        self.changes = ComplexChanges()

    def _set_changed_options(self):
        changed = {}
        for key in ComplexParameters.returnables:
            if getattr(self.want, key) is not None:
                changed[key] = getattr(self.want, key)
        if changed:
            self.changes = ComplexChanges(params=changed)

    def _update_changed_options(self):
        diff = Difference(self.want, self.have)
        updatables = ComplexParameters.updatables
        changed = dict()
        for k in updatables:
            change = diff.compare(k)
            if change is None:
                continue
            else:
                changed[k] = change
        if changed:
            self.changes = ComplexChanges(params=changed)
            return True
        return False

    def exec_module(self):
        start = datetime.now().isoformat()
        version = tmos_version(self.client)
        changed = False
        result = dict()
        state = self.want.state
        if state in ['present', 'draft']:
            changed = self.present()
        elif state == 'absent':
            changed = self.absent()
        changes = self.changes.to_return()
        result.update(**changes)
        result.update(dict(changed=changed))
        send_teem(start, self.client, self.module, version)
        return result

    def should_update(self):
        result = self._update_changed_options()
        drafted = self.draft_status_changed()
        if any((x is True for x in [result, drafted])):
            return True
        return False

    def draft_status_changed(self):
        if self.draft_exists() and self.want.state == 'draft':
            drafted = False
        elif not self.draft_exists() and self.want.state == 'present':
            drafted = False
        else:
            drafted = True
        return drafted

    def present(self):
        if self.draft_exists() or self.policy_exists():
            return self.update()
        else:
            return self.create()

    def absent(self):
        changed = False
        if self.draft_exists() or self.policy_exists():
            changed = self.remove()
        return changed

    def remove(self):
        if self.module.check_mode:
            return True
        self.remove_from_device()
        if self.draft_exists() or self.policy_exists():
            raise F5ModuleError('Failed to delete the policy')
        return True

    def create(self):
        self._validate_creation_parameters()
        self._set_changed_options()
        if self.module.check_mode:
            return True
        if not self.draft_exists():
            self._create_new_policy_draft()
        self.update_on_device()
        if self.want.state == 'draft':
            return True
        else:
            return self.publish()

    def update(self):
        self.have = self.read_current_from_device()
        if not self.should_update():
            return False
        if self.module.check_mode:
            return True
        if not self.draft_exists():
            self._create_existing_policy_draft()
        if self._update_changed_options():
            self.update_on_device()
        if self.want.state == 'draft':
            return True
        else:
            return self.publish()

    def draft_exists(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name, sub_path='Drafts'))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError:
            return False
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            return False
        return True

    def policy_exists(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError:
            return False
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            return False
        return True

    def _create_existing_policy_draft(self):
        params = dict(createDraft=True)
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
        return True

    def _create_new_policy_draft(self):
        params = self.want.api_params()
        payload = dict(name=self.want.name, partition=self.want.partition, subPath='Drafts', **params)
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
        return True

    def update_on_device(self):
        params = self.changes.api_params()
        if params:
            uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name, sub_path='Drafts'))
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
        self._upsert_policy_rules_on_device(draft=True)

    def read_current_from_device(self):
        if self.draft_exists():
            uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name, sub_path='Drafts'))
        else:
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
        result = ComplexParameters(params=response)
        result.update(dict(rules=rules))
        return result

    def publish(self):
        params = dict(name=fq_name(self.want.partition, self.want.name, sub_path='Drafts'), command='publish')
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 403]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        return True

    def remove_policy_draft_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name, sub_path='Drafts'))
        response = self.client.api.delete(uri)
        if response.status == 200:
            return True
        raise F5ModuleError(response.content)

    def remove_policy_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        response = self.client.api.delete(uri)
        if response.status == 200:
            return True
        raise F5ModuleError(response.content)

    def remove_from_device(self):
        if self.draft_exists():
            self.remove_policy_draft_from_device()
        if self.policy_exists():
            self.remove_policy_from_device()
        return True