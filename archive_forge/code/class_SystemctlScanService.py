from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
class SystemctlScanService(BaseService):
    BAD_STATES = frozenset(['not-found', 'masked', 'failed'])

    def systemd_enabled(self):
        try:
            f = open('/proc/1/comm', 'r')
        except IOError:
            return False
        for line in f:
            if 'systemd' in line:
                return True
        return False

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

    def _list_from_unit_files(self, systemctl_path, services):
        rc, stdout, stderr = self.module.run_command('%s list-unit-files --no-pager --type service --all' % systemctl_path, use_unsafe_shell=True)
        if rc != 0:
            self.module.warn('Could not get unit files data from systemd: %s' % stderr)
        else:
            for line in [svc_line for svc_line in stdout.split('\n') if '.service' in svc_line]:
                try:
                    service_name, status_val = line.split()[:2]
                except IndexError:
                    self.module.fail_json(msg='Malformed output discovered from systemd list-unit-files: {0}'.format(line))
                if service_name not in services:
                    rc, stdout, stderr = self.module.run_command('%s show %s --property=ActiveState' % (systemctl_path, service_name), use_unsafe_shell=True)
                    state = 'unknown'
                    if not rc and stdout != '':
                        state = stdout.replace('ActiveState=', '').rstrip()
                    services[service_name] = {'name': service_name, 'state': state, 'status': status_val, 'source': 'systemd'}
                elif services[service_name]['status'] not in self.BAD_STATES:
                    services[service_name]['status'] = status_val

    def gather_services(self):
        services = {}
        if self.systemd_enabled():
            systemctl_path = self.module.get_bin_path('systemctl', opt_dirs=['/usr/bin', '/usr/local/bin'])
            if systemctl_path:
                self._list_from_units(systemctl_path, services)
                self._list_from_unit_files(systemctl_path, services)
        return services