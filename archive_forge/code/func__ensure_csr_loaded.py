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
def _ensure_csr_loaded(self):
    """Load the CSR into self.csr."""
    if self.csr is not None:
        return
    if self.csr_path is None and self.csr_content is None:
        return
    self.csr = load_certificate_request(path=self.csr_path, content=self.csr_content, backend=self.backend)