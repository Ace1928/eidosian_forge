from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_acme import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_entrust import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_ownca import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_selfsigned import (
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
class CertificateAbsent(OpenSSLObject):

    def __init__(self, module):
        super(CertificateAbsent, self).__init__(module.params['path'], module.params['state'], module.params['force'], module.check_mode)
        self.module = module
        self.return_content = module.params['return_content']
        self.backup = module.params['backup']
        self.backup_file = None

    def generate(self, module):
        pass

    def remove(self, module):
        if self.backup:
            self.backup_file = module.backup_local(self.path)
        super(CertificateAbsent, self).remove(module)

    def dump(self, check_mode=False):
        result = {'changed': self.changed, 'filename': self.path, 'privatekey': self.module.params['privatekey_path'], 'csr': self.module.params['csr_path']}
        if self.backup_file:
            result['backup_file'] = self.backup_file
        if self.return_content:
            result['certificate'] = None
        return result