from __future__ import (absolute_import, division, print_function)
import copy
import os
import os.path
import re
import tempfile
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleFileNotFound, AnsibleParserError
from ansible.module_utils.basic import is_executable
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.quoting import unquote
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.vault import VaultLib, b_HEADER, is_encrypted, is_encrypted_file, parse_vaulttext_envelope, PromptVaultSecret
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
def _is_role(self, path: str) -> bool:
    """ imperfect role detection, roles are still valid w/o tasks|meta/main.yml|yaml|etc """
    b_path = to_bytes(path, errors='surrogate_or_strict')
    b_path_dirname = os.path.dirname(b_path)
    b_upath = to_bytes(unfrackpath(path, follow=False), errors='surrogate_or_strict')
    untasked_paths = (os.path.join(b_path, b'main.yml'), os.path.join(b_path, b'main.yaml'), os.path.join(b_path, b'main'))
    tasked_paths = (os.path.join(b_upath, b'tasks/main.yml'), os.path.join(b_upath, b'tasks/main.yaml'), os.path.join(b_upath, b'tasks/main'), os.path.join(b_upath, b'meta/main.yml'), os.path.join(b_upath, b'meta/main.yaml'), os.path.join(b_upath, b'meta/main'), os.path.join(b_path_dirname, b'tasks/main.yml'), os.path.join(b_path_dirname, b'tasks/main.yaml'), os.path.join(b_path_dirname, b'tasks/main'), os.path.join(b_path_dirname, b'meta/main.yml'), os.path.join(b_path_dirname, b'meta/main.yaml'), os.path.join(b_path_dirname, b'meta/main'))
    exists_untasked = map(os.path.exists, untasked_paths)
    exists_tasked = map(os.path.exists, tasked_paths)
    if RE_TASKS.search(path) and any(exists_untasked) or any(exists_tasked):
        return True
    return False