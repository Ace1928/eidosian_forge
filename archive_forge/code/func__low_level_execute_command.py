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
def _low_level_execute_command(self, cmd, sudoable=True, in_data=None, executable=None, encoding_errors='surrogate_then_replace', chdir=None):
    """
        This is the function which executes the low level shell command, which
        may be commands to create/remove directories for temporary files, or to
        run the module code or python directly when pipelining.

        :kwarg encoding_errors: If the value returned by the command isn't
            utf-8 then we have to figure out how to transform it to unicode.
            If the value is just going to be displayed to the user (or
            discarded) then the default of 'replace' is fine.  If the data is
            used as a key or is going to be written back out to a file
            verbatim, then this won't work.  May have to use some sort of
            replacement strategy (python3 could use surrogateescape)
        :kwarg chdir: cd into this directory before executing the command.
        """
    display.debug('_low_level_execute_command(): starting')
    if chdir:
        display.debug('_low_level_execute_command(): changing cwd to %s for this command' % chdir)
        cmd = self._connection._shell.append_command('cd %s' % chdir, cmd)
    if executable:
        self._connection._shell.executable = executable
    ruser = self._get_remote_user()
    buser = self.get_become_option('become_user')
    if sudoable and self._connection.become and (resource_from_fqcr(self._connection.transport) != 'network_cli') and (C.BECOME_ALLOW_SAME_USER or (buser != ruser or not any((ruser, buser)))):
        display.debug('_low_level_execute_command(): using become for this command')
        cmd = self._connection.become.build_become_command(cmd, self._connection._shell)
    if self._connection.allow_executable:
        if executable is None:
            executable = self._play_context.executable
            cmd = self._connection._shell.append_command(cmd, 'sleep 0')
        if executable:
            cmd = executable + ' -c ' + shlex.quote(cmd)
    display.debug('_low_level_execute_command(): executing: %s' % (cmd,))
    if self._connection.transport == 'local':
        self._connection.cwd = to_bytes(self._loader.get_basedir(), errors='surrogate_or_strict')
    rc, stdout, stderr = self._connection.exec_command(cmd, in_data=in_data, sudoable=sudoable)
    if isinstance(stdout, binary_type):
        out = to_text(stdout, errors=encoding_errors)
    elif not isinstance(stdout, text_type):
        out = to_text(b''.join(stdout.readlines()), errors=encoding_errors)
    else:
        out = stdout
    if isinstance(stderr, binary_type):
        err = to_text(stderr, errors=encoding_errors)
    elif not isinstance(stderr, text_type):
        err = to_text(b''.join(stderr.readlines()), errors=encoding_errors)
    else:
        err = stderr
    if rc is None:
        rc = 0
    out = self._strip_success_message(out)
    display.debug(u'_low_level_execute_command() done: rc=%d, stdout=%s, stderr=%s' % (rc, out, err))
    return dict(rc=rc, stdout=out, stdout_lines=out.splitlines(), stderr=err, stderr_lines=err.splitlines())