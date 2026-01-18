from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey import (
class PrivateKeyModule(OpenSSLObject):

    def __init__(self, module, module_backend):
        super(PrivateKeyModule, self).__init__(module.params['path'], module.params['state'], module.params['force'], module.check_mode)
        self.module_backend = module_backend
        self.return_content = module.params['return_content']
        if self.force:
            module_backend.regenerate = 'always'
        self.backup = module.params['backup']
        self.backup_file = None
        if module.params['mode'] is None:
            module.params['mode'] = '0600'
        module_backend.set_existing(load_file_if_exists(self.path, module))

    def generate(self, module):
        """Generate a keypair."""
        if self.module_backend.needs_regeneration():
            if not self.check_mode:
                if self.backup:
                    self.backup_file = module.backup_local(self.path)
                self.module_backend.generate_private_key()
                privatekey_data = self.module_backend.get_private_key_data()
                if self.return_content:
                    self.privatekey_bytes = privatekey_data
                write_file(module, privatekey_data, 384)
            self.changed = True
        elif self.module_backend.needs_conversion():
            if not self.check_mode:
                if self.backup:
                    self.backup_file = module.backup_local(self.path)
                self.module_backend.convert_private_key()
                privatekey_data = self.module_backend.get_private_key_data()
                if self.return_content:
                    self.privatekey_bytes = privatekey_data
                write_file(module, privatekey_data, 384)
            self.changed = True
        file_args = module.load_file_common_arguments(module.params)
        if module.check_file_absent_if_check_mode(file_args['path']):
            self.changed = True
        else:
            self.changed = module.set_fs_attributes_if_different(file_args, self.changed)

    def remove(self, module):
        self.module_backend.set_existing(None)
        if self.backup and (not self.check_mode):
            self.backup_file = module.backup_local(self.path)
        super(PrivateKeyModule, self).remove(module)

    def dump(self):
        """Serialize the object into a dictionary."""
        result = self.module_backend.dump(include_key=self.return_content)
        result['filename'] = self.path
        result['changed'] = self.changed
        if self.backup_file:
            result['backup_file'] = self.backup_file
        return result