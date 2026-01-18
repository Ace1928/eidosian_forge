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
def _get_expected_values_ports(module, client, api_version, options, image, values, host_info):
    expected_values = {}
    if 'published_ports' in values:
        expected_bound_ports = {}
        for container_port, config in values['published_ports'].items():
            if isinstance(container_port, int):
                container_port = '%s/tcp' % container_port
            if len(config) == 1:
                if isinstance(config[0], int):
                    expected_bound_ports[container_port] = [{'HostIp': '0.0.0.0', 'HostPort': config[0]}]
                else:
                    expected_bound_ports[container_port] = [{'HostIp': config[0], 'HostPort': ''}]
            elif isinstance(config[0], tuple):
                expected_bound_ports[container_port] = []
                for host_ip, host_port in config:
                    expected_bound_ports[container_port].append({'HostIp': host_ip, 'HostPort': to_text(host_port, errors='surrogate_or_strict')})
            else:
                expected_bound_ports[container_port] = [{'HostIp': config[0], 'HostPort': to_text(config[1], errors='surrogate_or_strict')}]
        expected_values['published_ports'] = expected_bound_ports
    image_ports = []
    if image:
        image_exposed_ports = image['Config'].get('ExposedPorts') or {}
        image_ports = [_normalize_port(p) for p in image_exposed_ports]
    param_ports = []
    if 'ports' in values:
        param_ports = [to_text(p[0], errors='surrogate_or_strict') + '/' + p[1] for p in values['ports']]
    result = list(set(image_ports + param_ports))
    expected_values['exposed_ports'] = result
    if 'publish_all_ports' in values:
        expected_values['publish_all_ports'] = values['publish_all_ports']
    return expected_values