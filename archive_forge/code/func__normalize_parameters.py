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
def _normalize_parameters(self, thing, action=None, additional_args=None):
    """
        arguments can be fuzzy.  Deal with all the forms.
        """
    additional_args = {} if additional_args is None else additional_args
    final_args = dict()
    if additional_args:
        if isinstance(additional_args, string_types):
            templar = Templar(loader=None)
            if templar.is_template(additional_args):
                final_args['_variable_params'] = additional_args
            else:
                raise AnsibleParserError("Complex args containing variables cannot use bare variables (without Jinja2 delimiters), and must use the full variable style ('{{var_name}}')")
        elif isinstance(additional_args, dict):
            final_args.update(additional_args)
        else:
            raise AnsibleParserError('Complex args must be a dictionary or variable string ("{{var}}").')
    if action is not None:
        args = self._normalize_new_style_args(thing, action)
    else:
        action, args = self._normalize_old_style_args(thing)
        if args and 'args' in args:
            tmp_args = args.pop('args')
            if isinstance(tmp_args, string_types):
                tmp_args = parse_kv(tmp_args)
            args.update(tmp_args)
    if args and action not in FREEFORM_ACTIONS:
        for arg in args:
            arg = to_text(arg)
            if arg.startswith('_ansible_'):
                raise AnsibleError("invalid parameter specified for action '%s': '%s'" % (action, arg))
    if args:
        final_args.update(args)
    return (action, final_args)