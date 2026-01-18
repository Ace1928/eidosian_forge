from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
class ServiceScanService(BaseService):

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

    def gather_services(self):
        services = {}
        self.service_path = self.module.get_bin_path('service')
        self.chkconfig_path = self.module.get_bin_path('chkconfig')
        self.initctl_path = self.module.get_bin_path('initctl')
        self.rc_status_path = self.module.get_bin_path('rc-status')
        self.rc_update_path = self.module.get_bin_path('rc-update')
        if self.service_path and self.chkconfig_path is None and (self.rc_status_path is None):
            self._list_sysvinit(services)
        if self.initctl_path and self.chkconfig_path is None:
            self._list_upstart(services)
        elif self.chkconfig_path:
            self._list_rh(services)
        elif self.rc_status_path is not None and self.rc_update_path is not None:
            self._list_openrc(services)
        return services