from __future__ import (absolute_import, division, print_function)
import ansible.constants as C
from ansible.errors import AnsibleParserError, AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.splitter import parse_kv, split_args
from ansible.plugins.loader import module_loader, action_loader
from ansible.template import Templar
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.sentinel import Sentinel
def _split_module_string(self, module_string):
    """
        when module names are expressed like:
        action: copy src=a dest=b
        the first part of the string is the name of the module
        and the rest are strings pertaining to the arguments.
        """
    tokens = split_args(module_string)
    if len(tokens) > 1:
        return (tokens[0].strip(), ' '.join(tokens[1:]))
    else:
        return (tokens[0].strip(), '')