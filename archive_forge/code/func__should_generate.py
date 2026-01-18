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
def _should_generate(self):
    if self.original_private_key is None:
        return True
    elif self.regenerate == 'never':
        return False
    elif self.regenerate == 'fail':
        if not self._private_key_valid():
            self.module.fail_json(msg='Key has wrong type and/or size. Will not proceed. ' + 'To force regeneration, call the module with `generate` set to ' + '`partial_idempotence`, `full_idempotence` or `always`, or with `force=true`.')
        return False
    elif self.regenerate in ('partial_idempotence', 'full_idempotence'):
        return not self._private_key_valid()
    else:
        return True