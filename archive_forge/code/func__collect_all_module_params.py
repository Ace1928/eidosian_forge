from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _collect_all_module_params(self):
    all_module_options = set()
    for option, data in self.module.argument_spec.items():
        all_module_options.add(option)
        if 'aliases' in data:
            for alias in data['aliases']:
                all_module_options.add(alias)
    return all_module_options