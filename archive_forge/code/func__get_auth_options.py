from __future__ import absolute_import, division, print_function
import os
import re
import sys
import tempfile
import traceback
from contextlib import contextmanager
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
def _get_auth_options(self):
    options = dict()
    for key, value in self.client.auth_params.items():
        if value is not None:
            option = AUTH_PARAM_MAPPING.get(key)
            if option:
                options[option] = value
    return options