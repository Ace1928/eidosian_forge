from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _needs_host_info(self):
    for options, values in self.parameters:
        engine = options.get_engine(self.engine_driver.name)
        if engine.needs_host_info(values):
            return True
    return False