from __future__ import absolute_import, division, print_function
import os
import traceback
import base64
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class SignatureInfoBase(OpenSSLObject):

    def __init__(self, module, backend):
        super(SignatureInfoBase, self).__init__(path=module.params['path'], state='present', force=False, check_mode=module.check_mode)
        self.backend = backend
        self.signature = module.params['signature']
        self.certificate_path = module.params['certificate_path']
        self.certificate_content = module.params['certificate_content']
        if self.certificate_content is not None:
            self.certificate_content = self.certificate_content.encode('utf-8')

    def generate(self):
        pass

    def dump(self):
        pass