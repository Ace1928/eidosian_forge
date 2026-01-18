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
class GenericCertificate(OpenSSLObject):
    """Retrieve a certificate using the given module backend."""

    def __init__(self, module, module_backend):
        super(GenericCertificate, self).__init__(module.params['path'], module.params['state'], module.params['force'], module.check_mode)
        self.module = module
        self.return_content = module.params['return_content']
        self.backup = module.params['backup']
        self.backup_file = None
        self.module_backend = module_backend
        self.module_backend.set_existing(load_file_if_exists(self.path, module))

    def generate(self, module):
        if self.module_backend.needs_regeneration():
            if not self.check_mode:
                self.module_backend.generate_certificate()
                result = self.module_backend.get_certificate_data()
                if self.backup:
                    self.backup_file = module.backup_local(self.path)
                write_file(module, result)
            self.changed = True
        file_args = module.load_file_common_arguments(module.params)
        if module.check_file_absent_if_check_mode(file_args['path']):
            self.changed = True
        else:
            self.changed = module.set_fs_attributes_if_different(file_args, self.changed)

    def check(self, module, perms_required=True):
        """Ensure the resource is in its desired state."""
        return super(GenericCertificate, self).check(module, perms_required) and (not self.module_backend.needs_regeneration())

    def dump(self, check_mode=False):
        result = self.module_backend.dump(include_certificate=self.return_content)
        result.update({'changed': self.changed, 'filename': self.path})
        if self.backup_file:
            result['backup_file'] = self.backup_file
        return result