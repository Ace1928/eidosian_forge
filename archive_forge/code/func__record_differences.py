from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _record_differences(self, differences, options, param_values, engine, container, container_image, image, host_info):
    container_values = engine.get_value(self.module, container.raw, self.engine_driver.get_api_version(self.client), options.options, container_image, host_info)
    expected_values = engine.get_expected_values(self.module, self.client, self.engine_driver.get_api_version(self.client), options.options, image, param_values.copy(), host_info)
    for option in options.options:
        if option.name in expected_values:
            param_value = expected_values[option.name]
            container_value = container_values.get(option.name)
            match = engine.compare_value(option, param_value, container_value)
            if not match:
                if engine.ignore_mismatching_result(self.module, self.client, self.engine_driver.get_api_version(self.client), option, image, container_value, param_value):
                    continue
                p = param_value
                c = container_value
                if option.comparison_type == 'set':
                    if p is not None:
                        p = sorted(p)
                    if c is not None:
                        c = sorted(c)
                elif option.comparison_type == 'set(dict)':
                    if option.name == 'expected_mounts':

                        def sort_key_fn(x):
                            return x['target']
                    else:

                        def sort_key_fn(x):
                            return sorted(((a, to_text(b, errors='surrogate_or_strict')) for a, b in x.items()))
                    if p is not None:
                        p = sorted(p, key=sort_key_fn)
                    if c is not None:
                        c = sorted(c, key=sort_key_fn)
                differences.add(option.name, parameter=p, active=c)