from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
class K8sTaintAnsible:

    def __init__(self, module, client):
        self.module = module
        self.api_instance = core_v1_api.CoreV1Api(client.client)
        self.changed = False

    def get_node(self, name):
        try:
            node = self.api_instance.read_node(name=name)
        except ApiException as exc:
            if exc.reason == 'Not Found':
                self.module.fail_json(msg="Node '{0}' has not been found.".format(name))
            self.module.fail_json(msg="Failed to retrieve node '{0}' due to: {1}".format(name, exc.reason), status=exc.status)
        except Exception as exc:
            self.module.fail_json(msg="Failed to retrieve node '{0}' due to: {1}".format(name, to_native(exc)))
        return node

    def patch_node(self, taints):
        body = {'spec': {'taints': taints}}
        try:
            result = self.api_instance.patch_node(name=self.module.params.get('name'), body=body)
        except Exception as exc:
            self.module.fail_json(msg='Failed to patch node due to: {0}'.format(to_native(exc)))
        return result.to_dict()

    def execute_module(self):
        result = {'result': {}}
        state = self.module.params.get('state')
        taints = self.module.params.get('taints')
        name = self.module.params.get('name')
        node = self.get_node(name)
        existing_taints = node.spec.to_dict().get('taints') or []
        diff = _get_difference(taints, existing_taints)
        if state == 'present':
            if diff:
                self.changed = True
                if self.module.check_mode:
                    self.module.exit_json(changed=self.changed, **result)
                if self.module.params.get('replace'):
                    result['result'] = self.patch_node(taints=taints)
                    self.module.exit_json(changed=self.changed, **result)
                result['result'] = self.patch_node(taints=[*_get_difference(existing_taints, taints), *taints])
            elif _update_exists(existing_taints, taints):
                self.changed = True
                if self.module.check_mode:
                    self.module.exit_json(changed=self.changed, **result)
                result['result'] = self.patch_node(taints=[*_get_difference(existing_taints, taints), *taints])
            else:
                result['result'] = node.to_dict()
        elif state == 'absent':
            if not existing_taints:
                result['result'] = node.to_dict()
            if not diff:
                self.changed = True
                if self.module.check_mode:
                    self.module.exit_json(changed=self.changed, **result)
                self.patch_node(taints=_get_difference(existing_taints, taints))
            elif _get_intersection(existing_taints, taints):
                self.changed = True
                if self.module.check_mode:
                    self.module.exit_json(changed=self.changed, **result)
                self.patch_node(taints=_get_difference(existing_taints, taints))
            else:
                self.module.exit_json(changed=self.changed, **result)
        self.module.exit_json(changed=self.changed, **result)