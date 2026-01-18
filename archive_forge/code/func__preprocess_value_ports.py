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
def _preprocess_value_ports(module, client, api_version, options, values):
    if 'published_ports' not in values:
        return values
    found = False
    for port_spec in values['published_ports'].values():
        if port_spec[0] == _DEFAULT_IP_REPLACEMENT_STRING:
            found = True
            break
    if not found:
        return values
    default_ip = _get_default_host_ip(module, client)
    for port, port_spec in values['published_ports'].items():
        if port_spec[0] == _DEFAULT_IP_REPLACEMENT_STRING:
            values['published_ports'][port] = tuple([default_ip] + list(port_spec[1:]))
    return values