from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class RabbitMqFeatureFlag(object):

    def __init__(self, module, name, node):
        self.module = module
        self.name = name
        self.node = node
        self._rabbitmqctl = module.get_bin_path('rabbitmqctl', True)
        self.state = self.get_flag_state()

    def _exec(self, args, force_exec_in_check_mode=False):
        if not self.module.check_mode or (self.module.check_mode and force_exec_in_check_mode):
            cmd = [self._rabbitmqctl, '-q', '-n', self.node]
            rc, out, err = self.module.run_command(cmd + args, check_rc=True)
            return out.splitlines()
        return list()

    def get_flag_state(self):
        global_parameters = self._exec(['list_feature_flags'], True)
        for param_item in global_parameters:
            name, state = param_item.split('\t')
            if name == self.name:
                if state == 'enabled':
                    return 'enabled'
                return 'disabled'
        return 'unavailable'

    def enable(self):
        self._exec(['enable_feature_flag', self.name])