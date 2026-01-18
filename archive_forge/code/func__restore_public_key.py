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
@OpensshModule.trigger_change
@OpensshModule.skip_if_check_mode
def _restore_public_key(self):
    try:
        temp_public_key = self._create_temp_public_key(str(self._get_public_key()) + '\n')
        self._safe_secure_move([(temp_public_key, self.public_key_path)])
    except (IOError, OSError):
        self.module.fail_json(msg='The public key is missing or does not match the private key. ' + 'Unable to regenerate the public key.')
    if self.comment:
        self._update_comment()