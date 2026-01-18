import collections
from . import _constants as C
def _index_unknown_and_reserved_tokens(self, unknown_token, reserved_tokens):
    """Indexes unknown and reserved tokens."""
    self._unknown_token = unknown_token
    self._idx_to_token = [unknown_token]
    if reserved_tokens is None:
        self._reserved_tokens = None
    else:
        self._reserved_tokens = reserved_tokens[:]
        self._idx_to_token.extend(reserved_tokens)
    self._token_to_idx = {token: idx for idx, token in enumerate(self._idx_to_token)}