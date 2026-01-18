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
def _normalize_new_style_args(self, thing, action):
    """
        deals with fuzziness in new style module invocations
        accepting key=value pairs and dictionaries, and returns
        a dictionary of arguments

        possible example inputs:
            'echo hi', 'shell'
            {'region': 'xyz'}, 'ec2'
        standardized outputs like:
            { _raw_params: 'echo hi', _uses_shell: True }
        """
    if isinstance(thing, dict):
        args = thing
    elif isinstance(thing, string_types):
        check_raw = action in FREEFORM_ACTIONS
        args = parse_kv(thing, check_raw=check_raw)
    elif thing is None:
        args = None
    else:
        raise AnsibleParserError('unexpected parameter type in action: %s' % type(thing), obj=self._task_ds)
    return args