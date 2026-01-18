from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _collect_all_options(self, active_options):
    all_options = {}
    for options in active_options:
        for option in options.options:
            all_options[option.name] = option
    return all_options