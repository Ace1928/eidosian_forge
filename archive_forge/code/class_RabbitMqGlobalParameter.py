from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
class RabbitMqGlobalParameter(object):

    def __init__(self, module, name, value, node):
        self.module = module
        self.name = name
        self.value = value
        self.node = node
        self._value = None
        self._rabbitmqctl = module.get_bin_path('rabbitmqctl', True)

    def _exec(self, args, force_exec_in_check_mode=False):
        if not self.module.check_mode or (self.module.check_mode and force_exec_in_check_mode):
            cmd = [self._rabbitmqctl, '-q', '-n', self.node]
            rc, out, err = self.module.run_command(cmd + args, check_rc=True)
            return out.splitlines()
        return list()

    def get(self):
        global_parameters = [param for param in self._exec(['list_global_parameters'], True) if param.strip()]
        for idx, param_item in enumerate(global_parameters):
            name, value = param_item.split('\t')
            if idx == 0 and name == 'name' and (value == 'value'):
                continue
            if name == self.name:
                self._value = json.loads(value)
                return True
        return False

    def set(self):
        self._exec(['set_global_parameter', self.name, json.dumps(self.value)])

    def delete(self):
        self._exec(['clear_global_parameter', self.name])

    def has_modifications(self):
        return self.value != self._value