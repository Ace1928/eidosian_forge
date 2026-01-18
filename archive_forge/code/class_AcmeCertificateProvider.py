from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
class AcmeCertificateProvider(CertificateProvider):

    def validate_module_args(self, module):
        if module.params['acme_accountkey_path'] is None:
            module.fail_json(msg='The acme_accountkey_path option must be specified for the acme provider.')
        if module.params['acme_challenge_path'] is None:
            module.fail_json(msg='The acme_challenge_path option must be specified for the acme provider.')

    def needs_version_two_certs(self, module):
        return False

    def create_backend(self, module, backend):
        return AcmeCertificateBackend(module, backend)