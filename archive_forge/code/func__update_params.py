from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _update_params(self):
    if self.param_networks_cli_compatible is True and self.module.params['networks'] and (self.module.params['network_mode'] is None):
        self.module.params['network_mode'] = self.module.params['networks'][0]['name']
    if self.param_container_default_behavior == 'compatibility':
        old_default_values = dict(auto_remove=False, detach=True, init=False, interactive=False, memory='0', paused=False, privileged=False, read_only=False, tty=False)
        for param, value in old_default_values.items():
            if self.module.params[param] is None:
                self.module.params[param] = value