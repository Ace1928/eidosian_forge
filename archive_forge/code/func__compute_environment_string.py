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
def _compute_environment_string(self, raw_environment_out=None):
    """
        Builds the environment string to be used when executing the remote task.
        """
    final_environment = dict()
    if self._task.environment is not None:
        environments = self._task.environment
        if not isinstance(environments, list):
            environments = [environments]
        for environment in environments:
            if environment is None or len(environment) == 0:
                continue
            temp_environment = self._templar.template(environment)
            if not isinstance(temp_environment, dict):
                raise AnsibleError('environment must be a dictionary, received %s (%s)' % (temp_environment, type(temp_environment)))
            final_environment.update(temp_environment)
    if len(final_environment) > 0:
        final_environment = self._templar.template(final_environment)
    if isinstance(raw_environment_out, dict):
        raw_environment_out.clear()
        raw_environment_out.update(final_environment)
    return self._connection._shell.env_prefix(**final_environment)