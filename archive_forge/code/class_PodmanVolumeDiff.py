from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
class PodmanVolumeDiff:

    def __init__(self, module, info, podman_version):
        self.module = module
        self.version = podman_version
        self.default_dict = None
        self.info = lower_keys(info)
        self.params = self.defaultize()
        self.diff = {'before': {}, 'after': {}}
        self.non_idempotent = {}

    def defaultize(self):
        params_with_defaults = {}
        self.default_dict = PodmanVolumeDefaults(self.module, self.version).default_dict()
        for p in self.module.params:
            if self.module.params[p] is None and p in self.default_dict:
                params_with_defaults[p] = self.default_dict[p]
            else:
                params_with_defaults[p] = self.module.params[p]
        return params_with_defaults

    def _diff_update_and_compare(self, param_name, before, after):
        if before != after:
            self.diff['before'].update({param_name: before})
            self.diff['after'].update({param_name: after})
            return True
        return False

    def diffparam_label(self):
        before = self.info['labels'] if 'labels' in self.info else {}
        after = self.params['label']
        return self._diff_update_and_compare('label', before, after)

    def diffparam_driver(self):
        before = self.info['driver']
        after = self.params['driver']
        return self._diff_update_and_compare('driver', before, after)

    def diffparam_options(self):
        before = self.info['options'] if 'options' in self.info else {}
        before.pop('uid', None)
        before.pop('gid', None)
        before = ['='.join((k, v)) for k, v in before.items()]
        after = self.params['options']
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('options', before, after)

    def is_different(self):
        diff_func_list = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith('diffparam')]
        fail_fast = not bool(self.module._diff)
        different = False
        for func_name in diff_func_list:
            dff_func = getattr(self, func_name)
            if dff_func():
                if fail_fast:
                    return True
                else:
                    different = True
        for p in self.non_idempotent:
            if self.module.params[p] is not None and self.module.params[p] not in [{}, [], '']:
                different = True
        return different