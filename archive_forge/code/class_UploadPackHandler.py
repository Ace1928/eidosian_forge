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
class UploadPackHandler(PackHandler):
    """Protocol handler for uploading a pack to the client."""

    def __init__(self, backend, args, proto, stateless_rpc=False, advertise_refs=False) -> None:
        super().__init__(backend, proto, stateless_rpc=stateless_rpc)
        self.repo = backend.open_repository(args[0])
        self._graph_walker = None
        self.advertise_refs = advertise_refs
        self._processing_have_lines = False

    @classmethod
    def capabilities(cls):
        return [CAPABILITY_MULTI_ACK_DETAILED, CAPABILITY_MULTI_ACK, CAPABILITY_SIDE_BAND_64K, CAPABILITY_THIN_PACK, CAPABILITY_OFS_DELTA, CAPABILITY_NO_PROGRESS, CAPABILITY_INCLUDE_TAG, CAPABILITY_SHALLOW, CAPABILITY_NO_DONE]

    @classmethod
    def required_capabilities(cls):
        return (CAPABILITY_SIDE_BAND_64K, CAPABILITY_THIN_PACK, CAPABILITY_OFS_DELTA)

    def progress(self, message: bytes):
        pass

    def _start_pack_send_phase(self):
        if self.has_capability(CAPABILITY_SIDE_BAND_64K):
            if not self.has_capability(CAPABILITY_NO_PROGRESS):
                self.progress = partial(self.proto.write_sideband, SIDE_BAND_CHANNEL_PROGRESS)
            self.write_pack_data = partial(self.proto.write_sideband, SIDE_BAND_CHANNEL_DATA)
        else:
            self.write_pack_data = self.proto.write

    def get_tagged(self, refs=None, repo=None) -> Dict[ObjectID, ObjectID]:
        """Get a dict of peeled values of tags to their original tag shas.

        Args:
          refs: dict of refname -> sha of possible tags; defaults to all
            of the backend's refs.
          repo: optional Repo instance for getting peeled refs; defaults
            to the backend's repo, if available
        Returns: dict of peeled_sha -> tag_sha, where tag_sha is the sha of a
            tag whose peeled value is peeled_sha.
        """
        if not self.has_capability(CAPABILITY_INCLUDE_TAG):
            return {}
        if refs is None:
            refs = self.repo.get_refs()
        if repo is None:
            repo = getattr(self.repo, 'repo', None)
            if repo is None:
                return {}
        tagged = {}
        for name, sha in refs.items():
            peeled_sha = repo.get_peeled(name)
            if peeled_sha != sha:
                tagged[peeled_sha] = sha
        return tagged

    def handle(self):
        self._processing_have_lines = True
        graph_walker = _ProtocolGraphWalker(self, self.repo.object_store, self.repo.get_peeled, self.repo.refs.get_symrefs)
        wants = []

        def wants_wrapper(refs, **kwargs):
            wants.extend(graph_walker.determine_wants(refs, **kwargs))
            return wants
        missing_objects = self.repo.find_missing_objects(wants_wrapper, graph_walker, self.progress, get_tagged=self.get_tagged)
        object_ids = list(missing_objects)
        if len(wants) == 0:
            return
        if not graph_walker.handle_done(not self.has_capability(CAPABILITY_NO_DONE), self._done_received):
            return
        self._start_pack_send_phase()
        self.progress(('counting objects: %d, done.\n' % len(object_ids)).encode('ascii'))
        write_pack_from_container(self.write_pack_data, self.repo.object_store, object_ids)
        self.proto.write_pkt_line(None)