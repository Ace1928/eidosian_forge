from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _validate_pv(module, vg, pvs):
    """
    Function to validate if the physical volume (PV) is not already in use by
    another volume group or Oracle ASM.

    :param module: Ansible module argument spec.
    :param vg: Volume group name.
    :param pvs: Physical volume list.
    :return: [bool, message] or module.fail_json for errors.
    """
    lspv_cmd = module.get_bin_path('lspv', True)
    rc, current_lspv, stderr = module.run_command([lspv_cmd])
    if rc != 0:
        module.fail_json(msg="Failed executing 'lspv' command.", rc=rc, stdout=current_lspv, stderr=stderr)
    for pv in pvs:
        lspv_list = {}
        for line in current_lspv.splitlines():
            pv_data = line.split()
            lspv_list[pv_data[0]] = pv_data[2]
        if pv not in lspv_list.keys():
            module.fail_json(msg="Physical volume '%s' doesn't exist." % pv)
        if lspv_list[pv] == 'None':
            lquerypv_cmd = module.get_bin_path('lquerypv', True)
            rc, current_lquerypv, stderr = module.run_command([lquerypv_cmd, '-h', '/dev/%s' % pv, '20', '10'])
            if rc != 0:
                module.fail_json(msg='Failed executing lquerypv command.', rc=rc, stdout=current_lquerypv, stderr=stderr)
            if 'ORCLDISK' in current_lquerypv:
                module.fail_json("Physical volume '%s' is already used by Oracle ASM." % pv)
            msg = "Physical volume '%s' is ok to be used." % pv
            return (True, msg)
        elif vg != lspv_list[pv]:
            module.fail_json(msg="Physical volume '%s' is in use by another volume group '%s'." % (pv, lspv_list[pv]))
        msg = "Physical volume '%s' is already used by volume group '%s'." % (pv, lspv_list[pv])
        return (False, msg)