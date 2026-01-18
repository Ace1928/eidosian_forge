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
def _set_values_ports(module, data, api_version, options, values):
    if 'ports' in values:
        exposed_ports = {}
        for port_definition in values['ports']:
            port = port_definition
            proto = 'tcp'
            if isinstance(port_definition, tuple):
                if len(port_definition) == 2:
                    proto = port_definition[1]
                port = port_definition[0]
            exposed_ports['%s/%s' % (port, proto)] = {}
        data['ExposedPorts'] = exposed_ports
    if 'published_ports' in values:
        if 'HostConfig' not in data:
            data['HostConfig'] = {}
        data['HostConfig']['PortBindings'] = convert_port_bindings(values['published_ports'])
    if 'publish_all_ports' in values and values['publish_all_ports']:
        if 'HostConfig' not in data:
            data['HostConfig'] = {}
        data['HostConfig']['PublishAllPorts'] = values['publish_all_ports']