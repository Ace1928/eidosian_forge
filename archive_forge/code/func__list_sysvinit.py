from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def _list_sysvinit(self, services):
    rc, stdout, stderr = self.module.run_command('%s --status-all' % self.service_path)
    if rc == 4 and (not os.path.exists('/etc/init.d')):
        return
    if rc != 0:
        self.module.warn("Unable to query 'service' tool (%s): %s" % (rc, stderr))
    p = re.compile('^\\s*\\[ (?P<state>\\+|\\-) \\]\\s+(?P<name>.+)$', flags=re.M)
    for match in p.finditer(stdout):
        service_name = match.group('name')
        if match.group('state') == '+':
            service_state = 'running'
        else:
            service_state = 'stopped'
        services[service_name] = {'name': service_name, 'state': service_state, 'source': 'sysv'}