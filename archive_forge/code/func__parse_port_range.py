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
def _parse_port_range(range_or_port, module):
    """
    Parses a string containing either a single port or a range of ports.

    Returns a list of integers for each port in the list.
    """
    if '-' in range_or_port:
        try:
            start, end = [int(port) for port in range_or_port.split('-')]
        except Exception:
            module.fail_json(msg='Invalid port range: "{0}"'.format(range_or_port))
        if end < start:
            module.fail_json(msg='Invalid port range: "{0}"'.format(range_or_port))
        return list(range(start, end + 1))
    else:
        try:
            return [int(range_or_port)]
        except Exception:
            module.fail_json(msg='Invalid port: "{0}"'.format(range_or_port))