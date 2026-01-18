from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _rename_vg(self):
    """Renames the volume group."""
    vgrename_cmd = self.module.get_bin_path('vgrename', required=True)
    if self.module._diff:
        self.result['diff'] = {'before': {'vg': self.vg}, 'after': {'vg': self.vg_new}}
    if self.module.check_mode:
        self.result['msg'] = 'Running in check mode. The module would rename VG %s to %s.' % (self.vg, self.vg_new)
        self.result['changed'] = True
    else:
        vgrename_cmd_with_opts = [vgrename_cmd, self.vg, self.vg_new]
        dummy, vg_rename_out, dummy = self.module.run_command(vgrename_cmd_with_opts, check_rc=True)
        self.result['msg'] = vg_rename_out
        self.result['changed'] = True