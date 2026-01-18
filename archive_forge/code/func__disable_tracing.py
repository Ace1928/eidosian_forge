from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _disable_tracing(self):
    return self._exec(['trace_off', '-p', self.name])