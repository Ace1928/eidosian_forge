from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.facts.packages import CLIMgr
class PIP(CLIMgr):

    def __init__(self, pip, module):
        self.CLI = pip
        self.module = module

    def list_installed(self):
        rc, out, err = self.module.run_command([self._cli, 'list', '-l', '--format=json'])
        if rc != 0:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return json.loads(out)

    def get_package_details(self, package):
        package['source'] = self.CLI
        return package