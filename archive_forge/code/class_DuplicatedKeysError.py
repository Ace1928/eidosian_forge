from typing import Union
from huggingface_hub.utils import insecure_hashlib
class DuplicatedKeysError(Exception):
    """Raise an error when duplicate key found."""

    def __init__(self, key, duplicate_key_indices, fix_msg=''):
        self.key = key
        self.duplicate_key_indices = duplicate_key_indices
        self.fix_msg = fix_msg
        self.prefix = 'Found multiple examples generated with the same key'
        if len(duplicate_key_indices) <= 20:
            self.err_msg = f'\nThe examples at index {', '.join(duplicate_key_indices)} have the key {key}'
        else:
            self.err_msg = f'\nThe examples at index {', '.join(duplicate_key_indices[:20])}... ({len(duplicate_key_indices) - 20} more) have the key {key}'
        self.suffix = '\n' + fix_msg if fix_msg else ''
        super().__init__(f'{self.prefix}{self.err_msg}{self.suffix}')