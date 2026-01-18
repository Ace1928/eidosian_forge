from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _set_values_log(module, data, api_version, options, values):
    if 'log_driver' not in values:
        return
    log_config = {'Type': values['log_driver'], 'Config': values.get('log_options') or {}}
    if 'HostConfig' not in data:
        data['HostConfig'] = {}
    data['HostConfig']['LogConfig'] = log_config