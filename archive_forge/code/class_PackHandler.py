import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
class PackHandler(Handler):
    """Protocol handler for packs."""

    def __init__(self, backend, proto, stateless_rpc=False) -> None:
        super().__init__(backend, proto, stateless_rpc)
        self._client_capabilities: Optional[Set[bytes]] = None
        self._done_received = False

    @classmethod
    def capabilities(cls) -> Iterable[bytes]:
        raise NotImplementedError(cls.capabilities)

    @classmethod
    def innocuous_capabilities(cls) -> Iterable[bytes]:
        return [CAPABILITY_INCLUDE_TAG, CAPABILITY_THIN_PACK, CAPABILITY_NO_PROGRESS, CAPABILITY_OFS_DELTA, capability_agent()]

    @classmethod
    def required_capabilities(cls) -> Iterable[bytes]:
        """Return a list of capabilities that we require the client to have."""
        return []

    def set_client_capabilities(self, caps: Iterable[bytes]) -> None:
        allowable_caps = set(self.innocuous_capabilities())
        allowable_caps.update(self.capabilities())
        for cap in caps:
            if cap.startswith(CAPABILITY_AGENT + b'='):
                continue
            if cap not in allowable_caps:
                raise GitProtocolError('Client asked for capability %r that was not advertised.' % cap)
        for cap in self.required_capabilities():
            if cap not in caps:
                raise GitProtocolError('Client does not support required capability %r.' % cap)
        self._client_capabilities = set(caps)
        logger.info('Client capabilities: %s', caps)

    def has_capability(self, cap: bytes) -> bool:
        if self._client_capabilities is None:
            raise GitProtocolError('Server attempted to access capability %r before asking client' % cap)
        return cap in self._client_capabilities

    def notify_done(self) -> None:
        self._done_received = True