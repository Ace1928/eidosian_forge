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
def _execute_module(self, module_name=None, module_args=None, tmp=None, task_vars=None, persist_files=False, delete_remote_tmp=None, wrap_async=False):
    """
        Transfer and run a module along with its arguments.
        """
    if tmp is not None:
        display.warning('_execute_module no longer honors the tmp parameter. Action plugins should set self._connection._shell.tmpdir to share the tmpdir')
    del tmp
    if delete_remote_tmp is not None:
        display.warning('_execute_module no longer honors the delete_remote_tmp parameter. Action plugins should check self._connection._shell.tmpdir to see if a tmpdir existed before they were called to determine if they are responsible for removing it.')
    del delete_remote_tmp
    tmpdir = self._connection._shell.tmpdir
    if not self._is_pipelining_enabled('new', wrap_async) and tmpdir is None:
        self._make_tmp_path()
        tmpdir = self._connection._shell.tmpdir
    if task_vars is None:
        task_vars = dict()
    if module_name is None:
        module_name = self._task.action
    if module_args is None:
        module_args = self._task.args
    self._update_module_args(module_name, module_args, task_vars)
    remove_async_dir = None
    if wrap_async or self._task.async_val:
        async_dir = self.get_shell_option('async_dir', default='~/.ansible_async')
        remove_async_dir = len(self._task.environment)
        self._task.environment.append({'ANSIBLE_ASYNC_DIR': async_dir})
    module_style, shebang, module_data, module_path = self._configure_module(module_name=module_name, module_args=module_args, task_vars=task_vars)
    display.vvv('Using module file %s' % module_path)
    if not shebang and module_style != 'binary':
        raise AnsibleError('module (%s) is missing interpreter line' % module_name)
    self._used_interpreter = shebang
    remote_module_path = None
    if not self._is_pipelining_enabled(module_style, wrap_async):
        if tmpdir is None:
            self._make_tmp_path()
            tmpdir = self._connection._shell.tmpdir
        remote_module_filename = self._connection._shell.get_remote_filename(module_path)
        remote_module_path = self._connection._shell.join_path(tmpdir, 'AnsiballZ_%s' % remote_module_filename)
    args_file_path = None
    if module_style in ('old', 'non_native_want_json', 'binary'):
        args_file_path = self._connection._shell.join_path(tmpdir, 'args')
    if remote_module_path or module_style != 'new':
        display.debug('transferring module to remote %s' % remote_module_path)
        if module_style == 'binary':
            self._transfer_file(module_path, remote_module_path)
        else:
            self._transfer_data(remote_module_path, module_data)
        if module_style == 'old':
            args_data = ''
            for k, v in module_args.items():
                args_data += '%s=%s ' % (k, shlex.quote(text_type(v)))
            self._transfer_data(args_file_path, args_data)
        elif module_style in ('non_native_want_json', 'binary'):
            self._transfer_data(args_file_path, json.dumps(module_args))
        display.debug('done transferring module to remote')
    environment_string = self._compute_environment_string()
    if remove_async_dir is not None:
        del self._task.environment[remove_async_dir]
    remote_files = []
    if tmpdir and remote_module_path:
        remote_files = [tmpdir, remote_module_path]
    if args_file_path:
        remote_files.append(args_file_path)
    sudoable = True
    in_data = None
    cmd = ''
    if wrap_async and (not self._connection.always_pipeline_modules):
        async_module_style, shebang, async_module_data, async_module_path = self._configure_module(module_name='ansible.legacy.async_wrapper', module_args=dict(), task_vars=task_vars)
        async_module_remote_filename = self._connection._shell.get_remote_filename(async_module_path)
        remote_async_module_path = self._connection._shell.join_path(tmpdir, async_module_remote_filename)
        self._transfer_data(remote_async_module_path, async_module_data)
        remote_files.append(remote_async_module_path)
        async_limit = self._task.async_val
        async_jid = f'j{random.randint(0, 999999999999)}'
        interpreter = shebang.replace('#!', '').strip()
        async_cmd = [interpreter, remote_async_module_path, async_jid, async_limit, remote_module_path]
        if environment_string:
            async_cmd.insert(0, environment_string)
        if args_file_path:
            async_cmd.append(args_file_path)
        else:
            async_cmd.append('_')
        if not self._should_remove_tmp_path(tmpdir):
            async_cmd.append('-preserve_tmp')
        cmd = ' '.join((to_text(x) for x in async_cmd))
    else:
        if self._is_pipelining_enabled(module_style):
            in_data = module_data
            display.vvv('Pipelining is enabled.')
        else:
            cmd = remote_module_path
        cmd = self._connection._shell.build_module_command(environment_string, shebang, cmd, arg_path=args_file_path).strip()
    if remote_files:
        remote_files = [x for x in remote_files if x]
        self._fixup_perms2(remote_files, self._get_remote_user())
    res = self._low_level_execute_command(cmd, sudoable=sudoable, in_data=in_data)
    data = self._parse_returned_data(res)
    if data.pop('_ansible_suppress_tmpdir_delete', False):
        self._cleanup_remote_tmp = False
    if 'results' in data and (not isinstance(data['results'], Sequence) or isinstance(data['results'], string_types)):
        data['ansible_module_results'] = data['results']
        del data['results']
        display.warning("Found internal 'results' key in module return, renamed to 'ansible_module_results'.")
    remove_internal_keys(data)
    if wrap_async:
        self._connection._shell.tmpdir = None
        data['changed'] = True
    if 'stdout' in data and 'stdout_lines' not in data:
        txt = data.get('stdout', None) or u''
        data['stdout_lines'] = txt.splitlines()
    if 'stderr' in data and 'stderr_lines' not in data:
        txt = data.get('stderr', None) or u''
        data['stderr_lines'] = txt.splitlines()
    if self._discovered_interpreter_key:
        if data.get('ansible_facts') is None:
            data['ansible_facts'] = {}
        data['ansible_facts'][self._discovered_interpreter_key] = self._discovered_interpreter
    if self._discovery_warnings:
        if data.get('warnings') is None:
            data['warnings'] = []
        data['warnings'].extend(self._discovery_warnings)
    if self._discovery_deprecation_warnings:
        if data.get('deprecations') is None:
            data['deprecations'] = []
        data['deprecations'].extend(self._discovery_deprecation_warnings)
    data = wrap_var(data)
    display.debug('done with _execute_module (%s, %s)' % (module_name, module_args))
    return data