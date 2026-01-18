from __future__ import absolute_import, division, print_function
import abc
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
def _check_params_valid(self, module):
    """Check if the params are in the correct state"""
    try:
        with open(self.path, 'rb') as f:
            data = f.read()
        params = cryptography.hazmat.primitives.serialization.load_pem_parameters(data, backend=self.crypto_backend)
    except Exception as dummy:
        return False
    bits = count_bits(params.parameter_numbers().p)
    return bits == self.size