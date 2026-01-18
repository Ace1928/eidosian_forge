from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _validate_vg(module, vg):
    """
    Check the current state of volume group.

    :param module: Ansible module argument spec.
    :param vg: Volume Group name.
    :return: True (VG in varyon state) or False (VG in varyoff state) or
             None (VG does not exist), message.
    """
    lsvg_cmd = module.get_bin_path('lsvg', True)
    rc, current_active_vgs, err = module.run_command([lsvg_cmd, '-o'])
    if rc != 0:
        module.fail_json(msg="Failed executing '%s' command." % lsvg_cmd)
    rc, current_all_vgs, err = module.run_command([lsvg_cmd])
    if rc != 0:
        module.fail_json(msg="Failed executing '%s' command." % lsvg_cmd)
    if vg in current_all_vgs and vg not in current_active_vgs:
        msg = "Volume group '%s' is in varyoff state." % vg
        return (False, msg)
    if vg in current_active_vgs:
        msg = "Volume group '%s' is in varyon state." % vg
        return (True, msg)
    msg = "Volume group '%s' does not exist." % vg
    return (None, msg)