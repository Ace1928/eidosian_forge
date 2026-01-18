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
def _validate_key_load(self):
    if self._private_key_exists() and self.regenerate in ('never', 'fail', 'partial_idempotence') and (self.original_private_key is None or not self._private_key_readable()):
        self.module.fail_json(msg='Unable to read the key. The key is protected with a passphrase or broken. ' + 'Will not proceed. To force regeneration, call the module with `generate` ' + 'set to `full_idempotence` or `always`, or with `force=true`.')