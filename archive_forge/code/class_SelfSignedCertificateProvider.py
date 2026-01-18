from __future__ import absolute_import, division, print_function
import os
from random import randrange
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
class SelfSignedCertificateProvider(CertificateProvider):

    def validate_module_args(self, module):
        if module.params['privatekey_path'] is None and module.params['privatekey_content'] is None:
            module.fail_json(msg='One of privatekey_path and privatekey_content must be specified for the selfsigned provider.')

    def needs_version_two_certs(self, module):
        return module.params['selfsigned_version'] == 2

    def create_backend(self, module, backend):
        if backend == 'cryptography':
            return SelfSignedCertificateBackendCryptography(module)