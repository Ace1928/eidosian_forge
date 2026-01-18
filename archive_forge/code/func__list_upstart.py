from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def _list_upstart(self, services):
    p = re.compile('^\\s?(?P<name>.*)\\s(?P<goal>\\w+)\\/(?P<state>\\w+)(\\,\\sprocess\\s(?P<pid>[0-9]+))?\\s*$')
    rc, stdout, stderr = self.module.run_command('%s list' % self.initctl_path)
    if rc != 0:
        self.module.warn('Unable to query upstart for service data: %s' % stderr)
    else:
        real_stdout = stdout.replace('\r', '')
        for line in real_stdout.split('\n'):
            m = p.match(line)
            if not m:
                continue
            service_name = m.group('name')
            service_goal = m.group('goal')
            service_state = m.group('state')
            if m.group('pid'):
                pid = m.group('pid')
            else:
                pid = None
            payload = {'name': service_name, 'state': service_state, 'goal': service_goal, 'source': 'upstart'}
            services[service_name] = payload