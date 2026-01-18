from __future__ import (absolute_import, division, print_function)
import ast
import re
import warnings
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.utils.shlex import shlex_split
def _parse_host_definition(self, line):
    """
        Takes a single line and tries to parse it as a host definition. Returns
        a list of Hosts if successful, or raises an error.
        """
    try:
        tokens = shlex_split(line, comments=True)
    except ValueError as e:
        self._raise_error("Error parsing host definition '%s': %s" % (line, e))
    hostnames, port = self._expand_hostpattern(tokens[0])
    variables = {}
    for t in tokens[1:]:
        if '=' not in t:
            self._raise_error('Expected key=value host variable assignment, got: %s' % t)
        k, v = t.split('=', 1)
        variables[k] = self._parse_value(v)
    return (hostnames, port, variables)