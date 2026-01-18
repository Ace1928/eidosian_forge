import binascii
import logging
import os
import tempfile
from base64 import b64decode, b64encode
from datetime import timedelta
from typing import Any, Optional, Tuple, cast
from torch.distributed import FileStore, Store, TCPStore
from torch.distributed.elastic.events import (
from .api import (
from .dynamic_rendezvous import RendezvousBackend, Token
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint
class C10dRendezvousBackend(RendezvousBackend):
    """Represents a C10d-backed rendezvous backend.

    Args:
        store:
            The :py:class:`torch.distributed.Store` instance to use to
            communicate with the C10d store.
        run_id:
            The run id of the rendezvous.
    """
    _NULL_SENTINEL = 'Y2FuaW1hZGFt'
    _store: Store
    _key: str

    def __init__(self, store: Store, run_id: str) -> None:
        if not run_id:
            raise ValueError('The run id must be a non-empty string.')
        self._store = store
        self._key = 'torch.rendezvous.' + run_id
        self._call_store('compare_set', self._key, '', self._NULL_SENTINEL)

    @property
    def name(self) -> str:
        """See base class."""
        return 'c10d'

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        base64_state: bytes = self._call_store('get', self._key)
        return self._decode_state(base64_state)

    def set_state(self, state: bytes, token: Optional[Token]=None) -> Optional[Tuple[bytes, Token, bool]]:
        """See base class."""
        base64_state_str: str = b64encode(state).decode()
        if token:
            if not isinstance(token, bytes):
                result = self.get_state()
                if result is not None:
                    tmp = (*result, False)
                    return tmp
                return None
            token = token.decode()
        else:
            token = self._NULL_SENTINEL
        base64_state: bytes = self._call_store('compare_set', self._key, token, base64_state_str)
        state_token_pair = self._decode_state(base64_state)
        if state_token_pair is None:
            return None
        new_state, new_token = state_token_pair
        return (new_state, new_token, new_state == state)

    def _call_store(self, store_op: str, *args, **kwargs) -> Any:
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise RendezvousConnectionError('The connection to the C10d store has failed. See inner exception for details.') from exc

    def _decode_state(self, base64_state: bytes) -> Optional[Tuple[bytes, Token]]:
        if base64_state == self._NULL_SENTINEL.encode():
            return None
        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError('The state object is corrupt. See inner exception for details.') from exc
        return (state, base64_state)