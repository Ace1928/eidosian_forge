from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
class RpmOstreePkg:

    def __init__(self, module):
        self.module = module
        self.params = module.params
        self.state = module.params['state']

    def ensure(self):
        results = dict(rc=0, changed=False, action='', packages=[], stdout='', stderr='', cmd='')
        cmd = [self.module.get_bin_path('rpm-ostree', required=True)]
        if self.state in 'present':
            results['action'] = 'install'
            cmd.append('install')
        elif self.state in 'absent':
            results['action'] = 'uninstall'
            cmd.append('uninstall')
        cmd.extend(['--allow-inactive', '--idempotent', '--unchanged-exit-77'])
        for pkg in self.params['name']:
            cmd.append(pkg)
            results['packages'].append(pkg)
        rc, out, err = self.module.run_command(cmd)
        results.update(dict(rc=rc, cmd=' '.join(cmd), stdout=out, stderr=err))
        if rc == 0:
            results['changed'] = True
        elif rc == 77:
            results['changed'] = False
            results['rc'] = 0
        else:
            self.module.fail_json(msg='non-zero return code', **results)
        self.module.exit_json(**results)