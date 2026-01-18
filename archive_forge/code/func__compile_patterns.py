from __future__ import (absolute_import, division, print_function)
import ast
import re
import warnings
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.utils.shlex import shlex_split
def _compile_patterns(self):
    """
        Compiles the regular expressions required to parse the inventory and
        stores them in self.patterns.
        """
    self.patterns['section'] = re.compile(to_text('^\\[\n                    ([^:\\]\\s]+)             # group name (see groupname below)\n                    (?::(\\w+))?             # optional : and tag name\n                \\]\n                \\s*                         # ignore trailing whitespace\n                (?:\\#.*)?                   # and/or a comment till the\n                $                           # end of the line\n            ', errors='surrogate_or_strict'), re.X)
    self.patterns['groupname'] = re.compile(to_text('^\n                ([^:\\]\\s]+)\n                \\s*                         # ignore trailing whitespace\n                (?:\\#.*)?                   # and/or a comment till the\n                $                           # end of the line\n            ', errors='surrogate_or_strict'), re.X)