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
def _transfer_data(self, remote_path, data):
    """
        Copies the module data out to the temporary module path.
        """
    if isinstance(data, dict):
        data = jsonify(data)
    afd, afile = tempfile.mkstemp(dir=C.DEFAULT_LOCAL_TMP)
    afo = os.fdopen(afd, 'wb')
    try:
        data = to_bytes(data, errors='surrogate_or_strict')
        afo.write(data)
    except Exception as e:
        raise AnsibleError('failure writing module data to temporary file for transfer: %s' % to_native(e))
    afo.flush()
    afo.close()
    try:
        self._transfer_file(afile, remote_path)
    finally:
        os.unlink(afile)
    return remote_path