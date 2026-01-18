from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class Svc(object):
    """
    Main class that handles daemontools, can be subclassed and overridden in case
    we want to use a 'derivative' like encore, s6, etc
    """

    def __init__(self, module):
        self.extra_paths = ['/command', '/usr/local/bin']
        self.report_vars = ['state', 'enabled', 'downed', 'svc_full', 'src_full', 'pid', 'duration', 'full_state']
        self.module = module
        self.name = module.params['name']
        self.service_dir = module.params['service_dir']
        self.service_src = module.params['service_src']
        self.enabled = None
        self.downed = None
        self.full_state = None
        self.state = None
        self.pid = None
        self.duration = None
        self.svc_cmd = module.get_bin_path('svc', opt_dirs=self.extra_paths)
        self.svstat_cmd = module.get_bin_path('svstat', opt_dirs=self.extra_paths)
        self.svc_full = '/'.join([self.service_dir, self.name])
        self.src_full = '/'.join([self.service_src, self.name])
        self.enabled = os.path.lexists(self.svc_full)
        if self.enabled:
            self.downed = os.path.lexists('%s/down' % self.svc_full)
            self.get_status()
        else:
            self.downed = os.path.lexists('%s/down' % self.src_full)
            self.state = 'stopped'

    def enable(self):
        if os.path.exists(self.src_full):
            try:
                os.symlink(self.src_full, self.svc_full)
            except OSError as e:
                self.module.fail_json(path=self.src_full, msg='Error while linking: %s' % to_native(e))
        else:
            self.module.fail_json(msg='Could not find source for service to enable (%s).' % self.src_full)

    def disable(self):
        try:
            os.unlink(self.svc_full)
        except OSError as e:
            self.module.fail_json(path=self.svc_full, msg='Error while unlinking: %s' % to_native(e))
        self.execute_command([self.svc_cmd, '-dx', self.src_full])
        src_log = '%s/log' % self.src_full
        if os.path.exists(src_log):
            self.execute_command([self.svc_cmd, '-dx', src_log])

    def get_status(self):
        rc, out, err = self.execute_command([self.svstat_cmd, self.svc_full])
        if err is not None and err:
            self.full_state = self.state = err
        else:
            self.full_state = out
            m = re.search('\\(pid (\\d+)\\)', out)
            if m:
                self.pid = m.group(1)
            m = re.search('(\\d+) seconds', out)
            if m:
                self.duration = m.group(1)
            if re.search(' up ', out):
                self.state = 'start'
            elif re.search(' down ', out):
                self.state = 'stopp'
            else:
                self.state = 'unknown'
                return
            if re.search(' want ', out):
                self.state += 'ing'
            else:
                self.state += 'ed'

    def start(self):
        return self.execute_command([self.svc_cmd, '-u', self.svc_full])

    def stopp(self):
        return self.stop()

    def stop(self):
        return self.execute_command([self.svc_cmd, '-d', self.svc_full])

    def once(self):
        return self.execute_command([self.svc_cmd, '-o', self.svc_full])

    def reload(self):
        return self.execute_command([self.svc_cmd, '-1', self.svc_full])

    def restart(self):
        return self.execute_command([self.svc_cmd, '-t', self.svc_full])

    def kill(self):
        return self.execute_command([self.svc_cmd, '-k', self.svc_full])

    def execute_command(self, cmd):
        try:
            rc, out, err = self.module.run_command(cmd)
        except Exception as e:
            self.module.fail_json(msg='failed to execute: %s' % to_native(e), exception=traceback.format_exc())
        return (rc, out, err)

    def report(self):
        self.get_status()
        states = {}
        for k in self.report_vars:
            states[k] = self.__dict__[k]
        return states