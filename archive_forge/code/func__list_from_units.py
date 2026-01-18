from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def _list_from_units(self, systemctl_path, services):
    rc, stdout, stderr = self.module.run_command('%s list-units --no-pager --type service --all' % systemctl_path, use_unsafe_shell=True)
    if rc != 0:
        self.module.warn('Could not list units from systemd: %s' % stderr)
    else:
        for line in [svc_line for svc_line in stdout.split('\n') if '.service' in svc_line]:
            state_val = 'stopped'
            status_val = 'unknown'
            fields = line.split()
            for bad in self.BAD_STATES:
                if bad in fields:
                    status_val = bad
                    fields = fields[1:]
                    break
            else:
                status_val = fields[2]
            service_name = fields[0]
            if fields[3] == 'running':
                state_val = 'running'
            services[service_name] = {'name': service_name, 'state': state_val, 'status': status_val, 'source': 'systemd'}