from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _compare_time_parameters(self):
    try:
        original_time_parameters = OpensshCertificateTimeParameters(valid_from=self.original_data.valid_after, valid_to=self.original_data.valid_before)
    except ValueError as e:
        return self.module.fail_json(msg=to_native(e))
    if self.ignore_timestamps:
        return original_time_parameters.within_range(self.valid_at)
    return all([original_time_parameters == self.time_parameters, original_time_parameters.within_range(self.valid_at)])