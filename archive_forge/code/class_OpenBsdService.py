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
class OpenBsdService(Service):
    """
    This is the OpenBSD Service manipulation class - it uses rcctl(8) or
    /etc/rc.d scripts for service control. Enabling a service is
    only supported if rcctl is present.
    """
    platform = 'OpenBSD'
    distribution = None

    def get_service_tools(self):
        self.enable_cmd = self.module.get_bin_path('rcctl')
        if self.enable_cmd:
            self.svc_cmd = self.enable_cmd
        else:
            rcdir = '/etc/rc.d'
            rc_script = '%s/%s' % (rcdir, self.name)
            if os.path.isfile(rc_script):
                self.svc_cmd = rc_script
        if not self.svc_cmd:
            self.module.fail_json(msg='unable to find svc_cmd')

    def get_service_status(self):
        if self.enable_cmd:
            rc, stdout, stderr = self.execute_command('%s %s %s' % (self.svc_cmd, 'check', self.name))
        else:
            rc, stdout, stderr = self.execute_command('%s %s' % (self.svc_cmd, 'check'))
        if stderr:
            self.module.fail_json(msg=stderr)
        if rc == 1:
            self.running = False
        elif rc == 0:
            self.running = True

    def service_control(self):
        if self.enable_cmd:
            return self.execute_command('%s -f %s %s' % (self.svc_cmd, self.action, self.name), daemonize=True)
        else:
            return self.execute_command('%s -f %s' % (self.svc_cmd, self.action))

    def service_enable(self):
        if not self.enable_cmd:
            return super(OpenBsdService, self).service_enable()
        rc, stdout, stderr = self.execute_command('%s %s %s %s' % (self.enable_cmd, 'get', self.name, 'status'))
        status_action = None
        if self.enable:
            if rc != 0:
                status_action = 'on'
        elif self.enable is not None:
            if rc != 1:
                status_action = 'off'
        if status_action is not None:
            self.changed = True
            if not self.module.check_mode:
                rc, stdout, stderr = self.execute_command('%s set %s status %s' % (self.enable_cmd, self.name, status_action))
                if rc != 0:
                    if stderr:
                        self.module.fail_json(msg=stderr)
                    else:
                        self.module.fail_json(msg='rcctl failed to modify service status')