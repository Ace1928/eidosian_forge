from __future__ import (absolute_import, division, print_function)
import os
from collections.abc import MutableMapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.inventory import BaseFileInventoryPlugin
def _parse_host(self, host_pattern):
    """
        Each host key can be a pattern, try to process it and add variables as needed
        """
    try:
        hostnames, port = self._expand_hostpattern(host_pattern)
    except TypeError:
        raise AnsibleParserError(f'Host pattern {host_pattern} must be a string. Enclose integers/floats in quotation marks.')
    return (hostnames, port)