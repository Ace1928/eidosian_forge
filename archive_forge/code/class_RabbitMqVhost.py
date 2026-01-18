from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class RabbitMqVhost(object):

    def __init__(self, module, name, tracing, node):
        self.module = module
        self.name = name
        self.tracing = tracing
        self.node = node
        self._tracing = False
        self._rabbitmqctl = module.get_bin_path('rabbitmqctl', True)

    def _exec(self, args, force_exec_in_check_mode=False):
        if not self.module.check_mode or (self.module.check_mode and force_exec_in_check_mode):
            cmd = [self._rabbitmqctl, '-q', '-n', self.node]
            rc, out, err = self.module.run_command(cmd + args, check_rc=True)
            return out.splitlines()
        return list()

    def get(self):
        vhosts = self._exec(['list_vhosts', 'name', 'tracing'], True)
        for vhost in vhosts:
            if '\t' not in vhost:
                continue
            name, tracing = vhost.split('\t')
            if name == self.name:
                self._tracing = self.module.boolean(tracing)
                return True
        return False

    def add(self):
        return self._exec(['add_vhost', self.name])

    def delete(self):
        return self._exec(['delete_vhost', self.name])

    def set_tracing(self):
        if self.tracing != self._tracing:
            if self.tracing:
                self._enable_tracing()
            else:
                self._disable_tracing()
            return True
        return False

    def _enable_tracing(self):
        return self._exec(['trace_on', '-p', self.name])

    def _disable_tracing(self):
        return self._exec(['trace_off', '-p', self.name])