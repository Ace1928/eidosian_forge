from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class LvgRename(object):

    def __init__(self, module):
        """
        Orchestrates the lvg_rename module logic.

        :param module: An AnsibleModule instance.
        """
        self.module = module
        self.result = {'changed': False}
        self.vg_list = []
        self._load_params()

    def run(self):
        """Performs the module logic."""
        self._load_vg_list()
        old_vg_exists = self._is_vg_exists(vg=self.vg)
        new_vg_exists = self._is_vg_exists(vg=self.vg_new)
        if old_vg_exists:
            if new_vg_exists:
                self.module.fail_json(msg='The new VG name (%s) is already in use.' % self.vg_new)
            else:
                self._rename_vg()
        elif new_vg_exists:
            self.result['msg'] = 'The new VG (%s) already exists, nothing to do.' % self.vg_new
            self.module.exit_json(**self.result)
        else:
            self.module.fail_json(msg='Both current (%s) and new (%s) VG are missing.' % (self.vg, self.vg_new))
        self.module.exit_json(**self.result)

    def _load_params(self):
        """Load the parameters from the module."""
        self.vg = self.module.params['vg']
        self.vg_new = self.module.params['vg_new']

    def _load_vg_list(self):
        """Load the VGs from the system."""
        vgs_cmd = self.module.get_bin_path('vgs', required=True)
        vgs_cmd_with_opts = [vgs_cmd, '--noheadings', '--separator', ';', '-o', 'vg_name,vg_uuid']
        dummy, vg_raw_list, dummy = self.module.run_command(vgs_cmd_with_opts, check_rc=True)
        for vg_info in vg_raw_list.splitlines():
            vg_name, vg_uuid = vg_info.strip().split(';')
            self.vg_list.append(vg_name)
            self.vg_list.append(vg_uuid)

    def _is_vg_exists(self, vg):
        """
        Checks VG existence by name or UUID. It removes the '/dev/' prefix before checking.

        :param vg: A string with the name or UUID of the VG.
        :returns: A boolean indicates whether the VG exists or not.
        """
        vg_found = False
        dev_prefix = '/dev/'
        if vg.startswith(dev_prefix):
            vg_id = vg[len(dev_prefix):]
        else:
            vg_id = vg
        vg_found = vg_id in self.vg_list
        return vg_found

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