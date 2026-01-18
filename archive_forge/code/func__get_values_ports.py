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
def _get_values_ports(module, container, api_version, options, image, host_info):
    host_config = container['HostConfig']
    config = container['Config']
    if config.get('ExposedPorts') is not None:
        expected_exposed = [_normalize_port(p) for p in config.get('ExposedPorts', dict()).keys()]
    else:
        expected_exposed = []
    return {'published_ports': host_config.get('PortBindings'), 'exposed_ports': expected_exposed, 'publish_all_ports': host_config.get('PublishAllPorts')}