from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def _list_openrc(self, services):
    all_services_runlevels = {}
    rc, stdout, stderr = self.module.run_command("%s -a -s -m 2>&1 | grep '^ ' | tr -d '[]'" % self.rc_status_path, use_unsafe_shell=True)
    rc_u, stdout_u, stderr_u = self.module.run_command("%s show -v 2>&1 | grep '|'" % self.rc_update_path, use_unsafe_shell=True)
    for line in stdout_u.split('\n'):
        line_data = line.split('|')
        if len(line_data) < 2:
            continue
        service_name = line_data[0].strip()
        runlevels = line_data[1].strip()
        if not runlevels:
            all_services_runlevels[service_name] = None
        else:
            all_services_runlevels[service_name] = runlevels.split()
    for line in stdout.split('\n'):
        line_data = line.split()
        if len(line_data) < 2:
            continue
        service_name = line_data[0]
        service_state = line_data[1]
        service_runlevels = all_services_runlevels[service_name]
        service_data = {'name': service_name, 'runlevels': service_runlevels, 'state': service_state, 'source': 'openrc'}
        services[service_name] = service_data