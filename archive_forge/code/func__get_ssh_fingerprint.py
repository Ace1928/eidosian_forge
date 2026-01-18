from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ..module_utils.cloudstack import (
def _get_ssh_fingerprint(self, public_key):
    key = sshpubkeys.SSHKey(public_key)
    if hasattr(key, 'hash_md5'):
        return key.hash_md5().replace(to_native('MD5:'), to_native(''))
    return key.hash()