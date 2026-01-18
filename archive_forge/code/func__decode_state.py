import binascii
from base64 import b64decode, b64encode
from typing import Optional, Tuple, cast
import urllib3.exceptions  # type: ignore[import]
from etcd import Client as EtcdClient  # type: ignore[import]
from etcd import (
from torch.distributed import Store
from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
from .dynamic_rendezvous import RendezvousBackend, Token
from .etcd_store import EtcdStore
from .utils import parse_rendezvous_endpoint
def _decode_state(self, result: EtcdResult) -> Tuple[bytes, Token]:
    base64_state = result.value.encode()
    try:
        state = b64decode(base64_state)
    except binascii.Error as exc:
        raise RendezvousStateError('The state object is corrupt. See inner exception for details.') from exc
    return (state, result.modifiedIndex)