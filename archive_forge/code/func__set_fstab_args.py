from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def _set_fstab_args(fstab_file):
    result = []
    if fstab_file and fstab_file != '/etc/fstab' and (platform.system().lower() != 'sunos'):
        if platform.system().lower().endswith('bsd'):
            result.append('-F')
        else:
            result.append('-T')
        result.append(fstab_file)
    return result