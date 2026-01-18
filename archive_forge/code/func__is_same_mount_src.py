from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def _is_same_mount_src(module, src, mountpoint, linux_mounts):
    """Return True if the mounted fs on mountpoint is the same source than src. Return False if mountpoint is not a mountpoint"""
    if not ismount(mountpoint) and (not is_bind_mounted(module, linux_mounts, mountpoint)):
        return False
    if platform.system() == 'Linux' and linux_mounts is not None:
        if is_bind_mounted(module, linux_mounts, mountpoint, src):
            return True
    cmd = '%s -v' % module.get_bin_path('mount', required=True)
    rc, out, err = module.run_command(cmd)
    mounts = []
    if len(out):
        mounts = to_native(out).strip().split('\n')
    else:
        module.fail_json(msg="Unable to retrieve mount info with command '%s'" % cmd)
    for mnt in mounts:
        fields = mnt.split()
        mp_src = fields[0]
        mp_dst = fields[2]
        if mp_src == src and mp_dst == mountpoint:
            return True
    return False