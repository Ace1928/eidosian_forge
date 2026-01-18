from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _compose_update_parameters(self):
    result = {}
    for options, values in self.parameters:
        engine = options.get_engine(self.engine_driver.name)
        if not engine.can_update_value(self.engine_driver.get_api_version(self.client)):
            continue
        engine.update_value(self.module, result, self.engine_driver.get_api_version(self.client), options.options, values)
    return result