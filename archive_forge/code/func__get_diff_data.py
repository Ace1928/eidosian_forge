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
def _get_diff_data(self, destination, source, task_vars, content, source_file=True):
    diff = {}
    display.debug('Going to peek to see if file has changed permissions')
    peek_result = self._execute_module(module_name='ansible.legacy.file', module_args=dict(path=destination, _diff_peek=True), task_vars=task_vars, persist_files=True)
    if peek_result.get('failed', False):
        display.warning(u"Failed to get diff between '%s' and '%s': %s" % (os.path.basename(source), destination, to_text(peek_result.get(u'msg', u''))))
        return diff
    if peek_result.get('rc', 0) == 0:
        if peek_result.get('state') in (None, 'absent'):
            diff['before'] = u''
        elif peek_result.get('appears_binary'):
            diff['dst_binary'] = 1
        elif peek_result.get('size') and C.MAX_FILE_SIZE_FOR_DIFF > 0 and (peek_result['size'] > C.MAX_FILE_SIZE_FOR_DIFF):
            diff['dst_larger'] = C.MAX_FILE_SIZE_FOR_DIFF
        else:
            display.debug(u'Slurping the file %s' % source)
            dest_result = self._execute_module(module_name='ansible.legacy.slurp', module_args=dict(path=destination), task_vars=task_vars, persist_files=True)
            if 'content' in dest_result:
                dest_contents = dest_result['content']
                if dest_result['encoding'] == u'base64':
                    dest_contents = base64.b64decode(dest_contents)
                else:
                    raise AnsibleError('unknown encoding in content option, failed: %s' % to_native(dest_result))
                diff['before_header'] = destination
                diff['before'] = to_text(dest_contents)
        if source_file:
            st = os.stat(source)
            if C.MAX_FILE_SIZE_FOR_DIFF > 0 and st[stat.ST_SIZE] > C.MAX_FILE_SIZE_FOR_DIFF:
                diff['src_larger'] = C.MAX_FILE_SIZE_FOR_DIFF
            else:
                display.debug('Reading local copy of the file %s' % source)
                try:
                    with open(source, 'rb') as src:
                        src_contents = src.read()
                except Exception as e:
                    raise AnsibleError('Unexpected error while reading source (%s) for diff: %s ' % (source, to_native(e)))
                if b'\x00' in src_contents:
                    diff['src_binary'] = 1
                else:
                    if content:
                        diff['after_header'] = destination
                    else:
                        diff['after_header'] = source
                    diff['after'] = to_text(src_contents)
        else:
            display.debug(u'source of file passed in')
            diff['after_header'] = u'dynamically generated'
            diff['after'] = source
    if self._task.no_log:
        if 'before' in diff:
            diff['before'] = u''
        if 'after' in diff:
            diff['after'] = u" [[ Diff output has been hidden because 'no_log: true' was specified for this result ]]\n"
    return diff