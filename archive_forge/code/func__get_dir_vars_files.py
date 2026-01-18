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
def _get_dir_vars_files(self, path: str, extensions: list[str]) -> list[str]:
    found = []
    for spath in sorted(self.list_directory(path)):
        if not spath.startswith(u'.') and (not spath.endswith(u'~')):
            ext = os.path.splitext(spath)[-1]
            full_spath = os.path.join(path, spath)
            if self.is_directory(full_spath) and (not ext):
                found.extend(self._get_dir_vars_files(full_spath, extensions))
            elif self.is_file(full_spath) and (not ext or to_text(ext) in extensions):
                found.append(full_spath)
    return found