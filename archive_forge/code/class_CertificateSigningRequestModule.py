from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
class CertificateSigningRequestModule(object):

    def __init__(self, module, module_backend):
        self.check_mode = module.check_mode
        self.module = module
        self.module_backend = module_backend
        self.changed = False
        if module.params['content'] is not None:
            self.module_backend.set_existing(module.params['content'].encode('utf-8'))

    def generate(self, module):
        """Generate the certificate signing request."""
        if self.module_backend.needs_regeneration():
            if not self.check_mode:
                self.module_backend.generate_csr()
            else:
                self.module.deprecate('Check mode support for openssl_csr_pipe will change in community.crypto 3.0.0 to behave the same as without check mode. You can get that behavior right now by adding `check_mode: false` to the openssl_csr_pipe task. If you think this breaks your use-case of this module, please create an issue in the community.crypto repository', version='3.0.0', collection_name='community.crypto')
            self.changed = True

    def dump(self):
        """Serialize the object into a dictionary."""
        result = self.module_backend.dump(include_csr=True)
        result.update({'changed': self.changed})
        return result