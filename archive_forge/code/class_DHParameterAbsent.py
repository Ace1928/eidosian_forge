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
class DHParameterAbsent(DHParameterBase):

    def __init__(self, module):
        super(DHParameterAbsent, self).__init__(module)

    def _do_generate(self, module):
        """Actually generate the DH params."""
        pass

    def _check_params_valid(self, module):
        """Check if the params are in the correct state"""
        pass