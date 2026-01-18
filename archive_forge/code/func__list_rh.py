from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def _list_rh(self, services):
    p = re.compile('(?P<service>.*?)\\s+[0-9]:(?P<rl0>on|off)\\s+[0-9]:(?P<rl1>on|off)\\s+[0-9]:(?P<rl2>on|off)\\s+[0-9]:(?P<rl3>on|off)\\s+[0-9]:(?P<rl4>on|off)\\s+[0-9]:(?P<rl5>on|off)\\s+[0-9]:(?P<rl6>on|off)')
    rc, stdout, stderr = self.module.run_command('%s' % self.chkconfig_path, use_unsafe_shell=True)
    match_any = False
    for line in stdout.split('\n'):
        if p.match(line):
            match_any = True
    if not match_any:
        p_simple = re.compile('(?P<service>.*?)\\s+(?P<rl0>on|off)')
        match_any = False
        for line in stdout.split('\n'):
            if p_simple.match(line):
                match_any = True
        if match_any:
            rc, stdout, stderr = self.module.run_command('%s -l --allservices' % self.chkconfig_path, use_unsafe_shell=True)
        elif '--list' in stderr:
            rc, stdout, stderr = self.module.run_command('%s --list' % self.chkconfig_path, use_unsafe_shell=True)
    for line in stdout.split('\n'):
        m = p.match(line)
        if m:
            service_name = m.group('service')
            service_state = 'stopped'
            service_status = 'disabled'
            if m.group('rl3') == 'on':
                service_status = 'enabled'
            rc, stdout, stderr = self.module.run_command('%s %s status' % (self.service_path, service_name), use_unsafe_shell=True)
            service_state = rc
            if rc in (0,):
                service_state = 'running'
            else:
                output = stderr.lower()
                for x in ('root', 'permission', 'not in sudoers'):
                    if x in output:
                        self.module.warn('Insufficient permissions to query sysV service "%s" and their states' % service_name)
                        break
                else:
                    service_state = 'stopped'
            service_data = {'name': service_name, 'state': service_state, 'status': service_status, 'source': 'sysv'}
            services[service_name] = service_data