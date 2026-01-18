from __future__ import absolute_import, division, print_function
import glob
import json
import os
import platform
import re
import select
import shlex
import subprocess
import tempfile
import time
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.six import PY2, b
class SunOSService(Service):
    """
    This is the SunOS Service manipulation class - it uses the svcadm
    command for controlling services, and svcs command for checking status.
    It also tries to be smart about taking the service out of maintenance
    state if necessary.
    """
    platform = 'SunOS'
    distribution = None

    def get_service_tools(self):
        self.svcs_cmd = self.module.get_bin_path('svcs', True)
        if not self.svcs_cmd:
            self.module.fail_json(msg='unable to find svcs binary')
        self.svcadm_cmd = self.module.get_bin_path('svcadm', True)
        if not self.svcadm_cmd:
            self.module.fail_json(msg='unable to find svcadm binary')
        if self.svcadm_supports_sync():
            self.svcadm_sync = '-s'
        else:
            self.svcadm_sync = ''

    def svcadm_supports_sync(self):
        for line in open('/etc/release', 'r').readlines():
            m = re.match('\\s+Oracle Solaris (\\d+)\\.(\\d+).*', line.rstrip())
            if m and m.groups() >= ('11', '2'):
                return True

    def get_service_status(self):
        status = self.get_sunos_svcs_status()
        if status == 'online':
            self.running = True
        else:
            self.running = False

    def get_sunos_svcs_status(self):
        rc, stdout, stderr = self.execute_command('%s %s' % (self.svcs_cmd, self.name))
        if rc == 1:
            if stderr:
                self.module.fail_json(msg=stderr)
            else:
                self.module.fail_json(msg=stdout)
        lines = stdout.rstrip('\n').split('\n')
        status = lines[-1].split(' ')[0]
        return status

    def service_enable(self):
        rc, stdout, stderr = self.execute_command('%s -l %s' % (self.svcs_cmd, self.name))
        if rc != 0:
            if stderr:
                self.module.fail_json(msg=stderr)
            else:
                self.module.fail_json(msg=stdout)
        enabled = False
        temporary = False
        for line in stdout.split('\n'):
            if line.startswith('enabled'):
                if 'true' in line:
                    enabled = True
                if 'temporary' in line:
                    temporary = True
        startup_enabled = enabled and (not temporary) or (not enabled and temporary)
        if self.enable and startup_enabled:
            return
        elif not self.enable and (not startup_enabled):
            return
        if not self.module.check_mode:
            if self.enable:
                subcmd = 'enable -rs'
            else:
                subcmd = 'disable -s'
            rc, stdout, stderr = self.execute_command('%s %s %s' % (self.svcadm_cmd, subcmd, self.name))
            if rc != 0:
                if stderr:
                    self.module.fail_json(msg=stderr)
                else:
                    self.module.fail_json(msg=stdout)
        self.changed = True

    def service_control(self):
        status = self.get_sunos_svcs_status()
        if self.action in ['start', 'reload', 'restart'] and status in ['maintenance', 'degraded']:
            rc, stdout, stderr = self.execute_command('%s clear %s' % (self.svcadm_cmd, self.name))
            if rc != 0:
                return (rc, stdout, stderr)
            status = self.get_sunos_svcs_status()
        if status in ['maintenance', 'degraded']:
            self.module.fail_json(msg='Failed to bring service out of %s status.' % status)
        if self.action == 'start':
            subcmd = 'enable -rst'
        elif self.action == 'stop':
            subcmd = 'disable -st'
        elif self.action == 'reload':
            subcmd = 'refresh %s' % self.svcadm_sync
        elif self.action == 'restart' and status == 'online':
            subcmd = 'restart %s' % self.svcadm_sync
        elif self.action == 'restart' and status != 'online':
            subcmd = 'enable -rst'
        return self.execute_command('%s %s %s' % (self.svcadm_cmd, subcmd, self.name))