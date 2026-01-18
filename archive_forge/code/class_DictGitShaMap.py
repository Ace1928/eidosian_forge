import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
class DictGitShaMap(GitShaMap):
    """Git SHA map that uses a dictionary."""

    def __init__(self):
        self._by_sha = {}
        self._by_fileid = {}
        self._by_revid = {}

    def lookup_blob_id(self, fileid, revision):
        return self._by_fileid[revision][fileid]

    def lookup_git_sha(self, sha):
        if not isinstance(sha, bytes):
            raise TypeError(sha)
        yield from self._by_sha[sha].values()

    def lookup_tree_id(self, fileid, revision):
        return self._by_fileid[revision][fileid]

    def lookup_commit(self, revid):
        return self._by_revid[revid]

    def revids(self):
        for key, entries in self._by_sha.items():
            for type, type_data in entries.values():
                if type == 'commit':
                    yield type_data[0]

    def sha1s(self):
        return self._by_sha.keys()