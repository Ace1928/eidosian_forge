from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _set_time_parameters(self):
    try:
        self.time_parameters = OpensshCertificateTimeParameters(valid_from=self.module.params['valid_from'], valid_to=self.module.params['valid_to'])
    except ValueError as e:
        self.module.fail_json(msg=to_native(e))