from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def add_fcportsetmember(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'addfcportsetmember'
    cmdopts = {'portset': self.name, 'fcioportid': self.fcportid}
    self.changed = True
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('FCPortsetmember (%s) mapping is created with fcportid (%s) successfully.', self.name, self.fcportid)