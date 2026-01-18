from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_info import (
def _ensure_existing_certificate_loaded(self):
    """Load the existing certificate into self.existing_certificate."""
    if self.existing_certificate is not None:
        return
    if self.existing_certificate_bytes is None:
        return
    self.existing_certificate = load_certificate(path=None, content=self.existing_certificate_bytes, backend=self.backend)