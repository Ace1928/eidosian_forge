from __future__ import (absolute_import, division, print_function)
import ast
import re
import warnings
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.utils.shlex import shlex_split
def _parse_group_name(self, line):
    """
        Takes a single line and tries to parse it as a group name. Returns the
        group name if successful, or raises an error.
        """
    m = self.patterns['groupname'].match(line)
    if m:
        return m.group(1)
    self._raise_error('Expected group name, got: %s' % line)