from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
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