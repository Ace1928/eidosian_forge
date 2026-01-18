from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import random
import re
import shlex
import stat
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure, AnsibleActionSkip, AnsibleActionFail, AnsibleAuthenticationFailure
from ansible.executor.module_common import modify_module
from ansible.executor.interpreter_discovery import discover_interpreter, InterpreterDiscoveryRequiredError
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.module_utils.errors import UnsupportedError
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.utils.jsonify import jsonify
from ansible.release import __version__
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var, AnsibleUnsafeText
from ansible.vars.clean import remove_internal_keys
from ansible.utils.plugin_docs import get_versioned_doclink
def _parse_returned_data(self, res):
    try:
        filtered_output, warnings = _filter_non_json_lines(res.get('stdout', u''), objects_only=True)
        for w in warnings:
            display.warning(w)
        data = json.loads(filtered_output)
        if C.MODULE_STRICT_UTF8_RESPONSE and (not data.pop('_ansible_trusted_utf8', None)):
            try:
                _validate_utf8_json(data)
            except UnicodeEncodeError:
                display.deprecated(f'Module "{self._task.resolved_action or self._task.action}" returned non UTF-8 data in the JSON response. This will become an error in the future', version='2.18')
        data['_ansible_parsed'] = True
    except ValueError:
        data = dict(failed=True, _ansible_parsed=False)
        data['module_stdout'] = res.get('stdout', u'')
        if 'stderr' in res:
            data['module_stderr'] = res['stderr']
            if res['stderr'].startswith(u'Traceback'):
                data['exception'] = res['stderr']
        if 'exception' not in data and data['module_stdout'].startswith(u'Traceback'):
            data['exception'] = data['module_stdout']
        data['msg'] = 'MODULE FAILURE'
        if self._used_interpreter is not None:
            interpreter = re.escape(self._used_interpreter.lstrip('!#'))
            match = re.compile('%s: (?:No such file or directory|not found)' % interpreter)
            if match.search(data['module_stderr']) or match.search(data['module_stdout']):
                data['msg'] = 'The module failed to execute correctly, you probably need to set the interpreter.'
        data['msg'] += '\nSee stdout/stderr for the exact error'
        if 'rc' in res:
            data['rc'] = res['rc']
    return data