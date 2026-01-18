import sys
from dulwich.object_store import MissingObjectFinder, peel_sha
from dulwich.protocol import Protocol
from dulwich.server import (Backend, BackendRepo, ReceivePackHandler,
from .. import errors, trace
from ..controldir import ControlDir
from .mapping import decode_git_path, default_mapping
from .object_store import BazaarObjectStore, get_object_store
from .refs import get_refs_container
class BzrBackend(Backend):
    """A git serve backend that can use a Bazaar repository."""

    def __init__(self, transport):
        self.transport = transport
        self.mapping = default_mapping

    def open_repository(self, path):
        transport = self.transport.clone(decode_git_path(path).lstrip('/'))
        trace.mutter('client opens %r: %r', path, transport)
        return BzrBackendRepo(transport, self.mapping)