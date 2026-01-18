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
class FreeBsdService(Service):
    """
    This is the FreeBSD Service manipulation class - it uses the /etc/rc.conf
    file for controlling services started at boot and the 'service' binary to
    check status and perform direct service manipulation.
    """
    platform = 'FreeBSD'
    distribution = None

    def get_service_tools(self):
        self.svc_cmd = self.module.get_bin_path('service', True)
        if not self.svc_cmd:
            self.module.fail_json(msg='unable to find service binary')
        self.sysrc_cmd = self.module.get_bin_path('sysrc')

    def get_service_status(self):
        rc, stdout, stderr = self.execute_command('%s %s %s %s' % (self.svc_cmd, self.name, 'onestatus', self.arguments))
        if self.name == 'pf':
            self.running = 'Enabled' in stdout
        elif rc == 1:
            self.running = False
        elif rc == 0:
            self.running = True

    def service_enable(self):
        if self.enable:
            self.rcconf_value = 'YES'
        else:
            self.rcconf_value = 'NO'
        rcfiles = ['/etc/rc.conf', '/etc/rc.conf.local', '/usr/local/etc/rc.conf']
        for rcfile in rcfiles:
            if os.path.isfile(rcfile):
                self.rcconf_file = rcfile
        rc, stdout, stderr = self.execute_command('%s %s %s %s' % (self.svc_cmd, self.name, 'rcvar', self.arguments))
        try:
            rcvars = shlex.split(stdout, comments=True)
        except Exception:
            pass
        if not rcvars:
            self.module.fail_json(msg='unable to determine rcvar', stdout=stdout, stderr=stderr)
        for rcvar in rcvars:
            if '=' in rcvar:
                self.rcconf_key, default_rcconf_value = rcvar.split('=', 1)
                break
        if self.rcconf_key is None:
            self.module.fail_json(msg='unable to determine rcvar', stdout=stdout, stderr=stderr)
        if self.sysrc_cmd:
            rc, current_rcconf_value, stderr = self.execute_command('%s -n %s' % (self.sysrc_cmd, self.rcconf_key))
            if rc != 0:
                current_rcconf_value = default_rcconf_value
            if current_rcconf_value.strip().upper() != self.rcconf_value:
                self.changed = True
                if self.module.check_mode:
                    self.module.exit_json(changed=True, msg='changing service enablement')
                rc, change_stdout, change_stderr = self.execute_command('%s %s="%s"' % (self.sysrc_cmd, self.rcconf_key, self.rcconf_value))
                if rc != 0:
                    self.module.fail_json(msg='unable to set rcvar using sysrc', stdout=change_stdout, stderr=change_stderr)
                rc, check_stdout, check_stderr = self.execute_command('%s %s %s' % (self.svc_cmd, self.name, 'enabled'))
                if self.enable != (rc == 0):
                    self.module.fail_json(msg='unable to set rcvar: sysrc did not change value', stdout=change_stdout, stderr=change_stderr)
            else:
                self.changed = False
        else:
            try:
                return self.service_enable_rcconf()
            except Exception:
                self.module.fail_json(msg='unable to set rcvar')

    def service_control(self):
        if self.action == 'start':
            self.action = 'onestart'
        if self.action == 'stop':
            self.action = 'onestop'
        if self.action == 'reload':
            self.action = 'onereload'
        ret = self.execute_command('%s %s %s %s' % (self.svc_cmd, self.name, self.action, self.arguments))
        if self.sleep:
            time.sleep(self.sleep)
        return ret