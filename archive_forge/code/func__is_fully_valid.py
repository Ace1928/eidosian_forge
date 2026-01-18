from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _is_fully_valid(self):
    return self._is_partially_valid() and all([self._compare_options() if self.original_data.type == 'user' else True, self.original_data.key_id == self.identifier, self.original_data.public_key == self._get_key_fingerprint(self.public_key), self.original_data.signing_key == self._get_key_fingerprint(self.signing_key)])