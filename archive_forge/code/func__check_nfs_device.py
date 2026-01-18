from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils._mount import ismount
import re
def _check_nfs_device(module, nfs_host, device):
    """
    Validate if NFS server is exporting the device (remote export).

    :param module: Ansible module.
    :param nfs_host: nfs_host parameter, NFS server.
    :param device: device parameter, remote export.
    :return: True or False.
    """
    showmount_cmd = module.get_bin_path('showmount', True)
    rc, showmount_out, err = module.run_command([showmount_cmd, '-a', nfs_host])
    if rc != 0:
        module.fail_json(msg='Failed to run showmount. Error message: %s' % err)
    else:
        showmount_data = showmount_out.splitlines()
        for line in showmount_data:
            if line.split(':')[1] == device:
                return True
        return False