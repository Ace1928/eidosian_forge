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
class SqliteCacheUpdater(CacheUpdater):

    def __init__(self, cache, rev):
        self.cache = cache
        self.db = self.cache.idmap.db
        self.revid = rev.revision_id
        self._commit = None
        self._trees = []
        self._blobs = []

    def add_object(self, obj, bzr_key_data, path):
        if isinstance(obj, tuple):
            type_name, hexsha = obj
        else:
            type_name = obj.type_name.decode('ascii')
            hexsha = obj.id
        if not isinstance(hexsha, bytes):
            raise TypeError(hexsha)
        if type_name == 'commit':
            self._commit = obj
            if type(bzr_key_data) is not dict:
                raise TypeError(bzr_key_data)
            self._testament3_sha1 = bzr_key_data.get('testament3-sha1')
        elif type_name == 'tree':
            if bzr_key_data is not None:
                self._trees.append((hexsha, bzr_key_data[0], bzr_key_data[1]))
        elif type_name == 'blob':
            if bzr_key_data is not None:
                self._blobs.append((hexsha, bzr_key_data[0], bzr_key_data[1]))
        else:
            raise AssertionError

    def finish(self):
        if self._commit is None:
            raise AssertionError('No commit object added')
        self.db.executemany('replace into trees (sha1, fileid, revid) values (?, ?, ?)', self._trees)
        self.db.executemany('replace into blobs (sha1, fileid, revid) values (?, ?, ?)', self._blobs)
        self.db.execute('replace into commits (sha1, revid, tree_sha, testament3_sha1) values (?, ?, ?, ?)', (self._commit.id, self.revid, self._commit.tree, self._testament3_sha1))
        return self._commit