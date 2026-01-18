from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _compose_create_parameters(self, image):
    params = {}
    for options, values in self.parameters:
        engine = options.get_engine(self.engine_driver.name)
        if engine.can_set_value(self.engine_driver.get_api_version(self.client)):
            engine.set_value(self.module, params, self.engine_driver.get_api_version(self.client), options.options, values)
    params['Image'] = image
    return params