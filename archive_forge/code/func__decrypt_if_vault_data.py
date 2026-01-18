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
def _decrypt_if_vault_data(self, b_vault_data: bytes, b_file_name: bytes | None=None) -> tuple[bytes, bool]:
    """Decrypt b_vault_data if encrypted and return b_data and the show_content flag"""
    if not is_encrypted(b_vault_data):
        show_content = True
        return (b_vault_data, show_content)
    b_ciphertext, b_version, cipher_name, vault_id = parse_vaulttext_envelope(b_vault_data)
    b_data = self._vault.decrypt(b_vault_data, filename=b_file_name)
    show_content = False
    return (b_data, show_content)