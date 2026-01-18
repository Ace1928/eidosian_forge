from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def _generate_temp_keypair(self):
    temp_private_key = os.path.join(self.module.tmpdir, os.path.basename(self.private_key_path))
    temp_public_key = temp_private_key + '.pub'
    try:
        self._generate_keypair(temp_private_key)
    except (IOError, OSError) as e:
        self.module.fail_json(msg=to_native(e))
    for f in (temp_private_key, temp_public_key):
        self.module.add_cleanup_file(f)
    return (temp_private_key, temp_public_key)