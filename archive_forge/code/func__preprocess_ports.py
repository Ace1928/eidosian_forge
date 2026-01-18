from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _preprocess_ports(module, values):
    if 'published_ports' in values:
        if 'all' in values['published_ports']:
            module.fail_json(msg='Specifying "all" in published_ports is no longer allowed. Set publish_all_ports to "true" instead to randomly assign port mappings for those not specified by published_ports.')
        binds = {}
        for port in values['published_ports']:
            parts = _split_colon_ipv6(to_text(port, errors='surrogate_or_strict'), module)
            container_port = parts[-1]
            protocol = ''
            if '/' in container_port:
                container_port, protocol = parts[-1].split('/')
            container_ports = _parse_port_range(container_port, module)
            p_len = len(parts)
            if p_len == 1:
                port_binds = len(container_ports) * [(_DEFAULT_IP_REPLACEMENT_STRING,)]
            elif p_len == 2:
                if len(container_ports) == 1:
                    port_binds = [(_DEFAULT_IP_REPLACEMENT_STRING, parts[0])]
                else:
                    port_binds = [(_DEFAULT_IP_REPLACEMENT_STRING, port) for port in _parse_port_range(parts[0], module)]
            elif p_len == 3:
                ipaddr = parts[0]
                if not re.match('^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$', parts[0]) and (not re.match('^\\[[0-9a-fA-F:]+(?:|%[^\\]/]+)\\]$', ipaddr)):
                    module.fail_json(msg='Bind addresses for published ports must be IPv4 or IPv6 addresses, not hostnames. Use the dig lookup to resolve hostnames. (Found hostname: {0})'.format(ipaddr))
                if re.match('^\\[[0-9a-fA-F:]+\\]$', ipaddr):
                    ipaddr = ipaddr[1:-1]
                if parts[1]:
                    if len(container_ports) == 1:
                        port_binds = [(ipaddr, parts[1])]
                    else:
                        port_binds = [(ipaddr, port) for port in _parse_port_range(parts[1], module)]
                else:
                    port_binds = len(container_ports) * [(ipaddr,)]
            else:
                module.fail_json(msg='Invalid port description "%s" - expected 1 to 3 colon-separated parts, but got %d. Maybe you forgot to use square brackets ([...]) around an IPv6 address?' % (port, p_len))
            for bind, container_port in zip(port_binds, container_ports):
                idx = '{0}/{1}'.format(container_port, protocol) if protocol else container_port
                if idx in binds:
                    old_bind = binds[idx]
                    if isinstance(old_bind, list):
                        old_bind.append(bind)
                    else:
                        binds[idx] = [old_bind, bind]
                else:
                    binds[idx] = bind
        values['published_ports'] = binds
    exposed = []
    if 'exposed_ports' in values:
        for port in values['exposed_ports']:
            port = to_text(port, errors='surrogate_or_strict').strip()
            protocol = 'tcp'
            match = re.search('(/.+$)', port)
            if match:
                protocol = match.group(1).replace('/', '')
                port = re.sub('/.+$', '', port)
            exposed.append((port, protocol))
    if 'published_ports' in values:
        for publish_port in values['published_ports']:
            match = False
            if isinstance(publish_port, string_types) and '/' in publish_port:
                port, protocol = publish_port.split('/')
                port = int(port)
            else:
                protocol = 'tcp'
                port = int(publish_port)
            for exposed_port in exposed:
                if exposed_port[1] != protocol:
                    continue
                if isinstance(exposed_port[0], string_types) and '-' in exposed_port[0]:
                    start_port, end_port = exposed_port[0].split('-')
                    if int(start_port) <= port <= int(end_port):
                        match = True
                elif exposed_port[0] == port:
                    match = True
            if not match:
                exposed.append((port, protocol))
    values['ports'] = exposed
    return values