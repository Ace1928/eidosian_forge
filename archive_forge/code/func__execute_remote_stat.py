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
def _execute_remote_stat(self, path, all_vars, follow, tmp=None, checksum=True):
    """
        Get information from remote file.
        """
    if tmp is not None:
        display.warning('_execute_remote_stat no longer honors the tmp parameter. Action plugins should set self._connection._shell.tmpdir to share the tmpdir')
    del tmp
    module_args = dict(path=path, follow=follow, get_checksum=checksum, checksum_algorithm='sha1')
    mystat = self._execute_module(module_name='ansible.legacy.stat', module_args=module_args, task_vars=all_vars, wrap_async=False)
    if mystat.get('failed'):
        msg = mystat.get('module_stderr')
        if not msg:
            msg = mystat.get('module_stdout')
        if not msg:
            msg = mystat.get('msg')
        raise AnsibleError('Failed to get information on remote file (%s): %s' % (path, msg))
    if not mystat['stat']['exists']:
        mystat['stat']['checksum'] = '1'
    if 'checksum' not in mystat['stat']:
        mystat['stat']['checksum'] = ''
    elif not isinstance(mystat['stat']['checksum'], string_types):
        raise AnsibleError('Invalid checksum returned by stat: expected a string type but got %s' % type(mystat['stat']['checksum']))
    return mystat['stat']