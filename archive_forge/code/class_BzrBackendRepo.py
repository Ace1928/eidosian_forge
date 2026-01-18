import sys
from dulwich.object_store import MissingObjectFinder, peel_sha
from dulwich.protocol import Protocol
from dulwich.server import (Backend, BackendRepo, ReceivePackHandler,
from .. import errors, trace
from ..controldir import ControlDir
from .mapping import decode_git_path, default_mapping
from .object_store import BazaarObjectStore, get_object_store
from .refs import get_refs_container
class BzrBackendRepo(BackendRepo):

    def __init__(self, transport, mapping):
        self.mapping = mapping
        self.repo_dir = ControlDir.open_from_transport(transport)
        self.repo = self.repo_dir.find_repository()
        self.object_store = get_object_store(self.repo)
        self.refs = get_refs_container(self.repo_dir, self.object_store)

    def get_refs(self):
        with self.object_store.lock_read():
            return self.refs.as_dict()

    def get_peeled(self, name):
        cached = self.refs.get_peeled(name)
        if cached is not None:
            return cached
        return peel_sha(self.object_store, self.refs[name])[1].id

    def find_missing_objects(self, determine_wants, graph_walker, progress, get_tagged=None):
        """Yield git objects to send to client """
        with self.object_store.lock_read():
            wants = determine_wants(self.get_refs())
            have = self.object_store.find_common_revisions(graph_walker)
            if wants is None:
                return
            shallows = getattr(graph_walker, 'shallow', frozenset())
            if isinstance(self.object_store, BazaarObjectStore):
                return self.object_store.find_missing_objects(have, wants, shallow=shallows, progress=progress, get_tagged=get_tagged, lossy=True)
            else:
                return MissingObjectFinder(self.object_store, have, wants, shallow=shallows, progress=progress)