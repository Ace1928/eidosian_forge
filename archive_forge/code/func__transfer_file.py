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
def _transfer_file(self, local_path, remote_path):
    """
        Copy a file from the controller to a remote path

        :arg local_path: Path on controller to transfer
        :arg remote_path: Path on the remote system to transfer into

        .. warning::
            * When you use this function you likely want to use use fixup_perms2() on the
              remote_path to make sure that the remote file is readable when the user becomes
              a non-privileged user.
            * If you use fixup_perms2() on the file and copy or move the file into place, you will
              need to then remove filesystem acls on the file once it has been copied into place by
              the module.  See how the copy module implements this for help.
        """
    self._connection.put_file(local_path, remote_path)
    return remote_path