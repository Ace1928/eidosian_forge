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
class IndexCacheUpdater(CacheUpdater):

    def __init__(self, cache, rev):
        self.cache = cache
        self.revid = rev.revision_id
        self.parent_revids = rev.parent_ids
        self._commit = None
        self._entries = []

    def add_object(self, obj, bzr_key_data, path):
        if isinstance(obj, tuple):
            type_name, hexsha = obj
        else:
            type_name = obj.type_name.decode('ascii')
            hexsha = obj.id
        if type_name == 'commit':
            self._commit = obj
            if type(bzr_key_data) is not dict:
                raise TypeError(bzr_key_data)
            self.cache.idmap._add_git_sha(hexsha, b'commit', (self.revid, obj.tree, bzr_key_data))
            self.cache.idmap._add_node((b'commit', self.revid, b'X'), b' '.join((hexsha, obj.tree)))
        elif type_name == 'blob':
            self.cache.idmap._add_git_sha(hexsha, b'blob', bzr_key_data)
            self.cache.idmap._add_node((b'blob', bzr_key_data[0], bzr_key_data[1]), hexsha)
        elif type_name == 'tree':
            self.cache.idmap._add_git_sha(hexsha, b'tree', bzr_key_data)
        else:
            raise AssertionError

    def finish(self):
        return self._commit