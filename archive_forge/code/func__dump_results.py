from __future__ import (absolute_import, division, print_function)
import difflib
import json
import re
import sys
import textwrap
from collections import OrderedDict
from collections.abc import MutableMapping
from copy import deepcopy
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import text_type
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.parsing.yaml.objects import AnsibleUnicode
from ansible.plugins import AnsiblePlugin
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import AnsibleUnsafeText, NativeJinjaUnsafeText, _is_unsafe
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
import yaml
def _dump_results(self, result, indent=None, sort_keys=True, keep_invocation=False, serialize=True):
    try:
        result_format = self.get_option('result_format')
    except KeyError:
        result_format = 'json'
    try:
        pretty_results = self.get_option('pretty_results')
    except KeyError:
        pretty_results = None
    indent_conditions = (result.get('_ansible_verbose_always'), pretty_results is None and result_format != 'json', pretty_results is True, self._display.verbosity > 2)
    if not indent and any(indent_conditions):
        indent = 4
    if pretty_results is False:
        indent = None
    abridged_result = strip_internal_keys(module_response_deepcopy(result))
    if not keep_invocation and self._display.verbosity < 3 and ('invocation' in result):
        del abridged_result['invocation']
    if self._display.verbosity < 3 and 'diff' in result:
        del abridged_result['diff']
    if 'exception' in abridged_result:
        del abridged_result['exception']
    if not serialize:
        return abridged_result
    if result_format == 'json':
        try:
            return json.dumps(abridged_result, cls=AnsibleJSONEncoder, indent=indent, ensure_ascii=False, sort_keys=sort_keys)
        except TypeError:
            if not OrderedDict:
                raise
            return json.dumps(OrderedDict(sorted(abridged_result.items(), key=to_text)), cls=AnsibleJSONEncoder, indent=indent, ensure_ascii=False, sort_keys=False)
    elif result_format == 'yaml':
        lossy = pretty_results in (None, True)
        if lossy:
            if 'stdout' in abridged_result and 'stdout_lines' in abridged_result:
                abridged_result['stdout_lines'] = '<omitted>'
            if 'stderr' in abridged_result and 'stderr_lines' in abridged_result:
                abridged_result['stderr_lines'] = '<omitted>'
        return '\n%s' % textwrap.indent(yaml.dump(abridged_result, allow_unicode=True, Dumper=_AnsibleCallbackDumper(lossy=lossy), default_flow_style=False, indent=indent), ' ' * (indent or 4))