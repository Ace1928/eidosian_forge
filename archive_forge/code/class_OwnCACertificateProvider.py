from __future__ import absolute_import, division, print_function
import os
from random import randrange
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
class OwnCACertificateProvider(CertificateProvider):

    def validate_module_args(self, module):
        if module.params['ownca_path'] is None and module.params['ownca_content'] is None:
            module.fail_json(msg='One of ownca_path and ownca_content must be specified for the ownca provider.')
        if module.params['ownca_privatekey_path'] is None and module.params['ownca_privatekey_content'] is None:
            module.fail_json(msg='One of ownca_privatekey_path and ownca_privatekey_content must be specified for the ownca provider.')

    def needs_version_two_certs(self, module):
        return module.params['ownca_version'] == 2

    def create_backend(self, module, backend):
        if backend == 'cryptography':
            return OwnCACertificateBackendCryptography(module)