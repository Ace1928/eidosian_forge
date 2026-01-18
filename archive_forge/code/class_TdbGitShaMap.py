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
class TdbGitShaMap(GitShaMap):
    """SHA Map that uses a TDB database.

    Entries:

    "git <sha1>" -> "<type> <type-data1> <type-data2>"
    "commit revid" -> "<sha1> <tree-id>"
    "tree fileid revid" -> "<sha1>"
    "blob fileid revid" -> "<sha1>"
    """
    TDB_MAP_VERSION = 3
    TDB_HASH_SIZE = 50000

    def __init__(self, path=None):
        import tdb
        self.path = path
        if path is None:
            self.db = {}
        else:
            if path not in mapdbs():
                mapdbs()[path] = tdb.Tdb(path, self.TDB_HASH_SIZE, tdb.DEFAULT, os.O_RDWR | os.O_CREAT)
            self.db = mapdbs()[path]
        try:
            if int(self.db[b'version']) not in (2, 3):
                trace.warning('SHA Map is incompatible (%s -> %d), rebuilding database.', self.db[b'version'], self.TDB_MAP_VERSION)
                self.db.clear()
        except KeyError:
            pass
        self.db[b'version'] = b'%d' % self.TDB_MAP_VERSION

    def start_write_group(self):
        """Start writing changes."""
        self.db.transaction_start()

    def commit_write_group(self):
        """Commit any pending changes."""
        self.db.transaction_commit()

    def abort_write_group(self):
        """Abort any pending changes."""
        self.db.transaction_cancel()

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.path)

    def lookup_commit(self, revid):
        try:
            return sha_to_hex(self.db[b'commit\x00' + revid][:20])
        except KeyError:
            raise KeyError('No cache entry for %r' % revid)

    def lookup_blob_id(self, fileid, revision):
        return sha_to_hex(self.db[b'\x00'.join((b'blob', fileid, revision))])

    def lookup_git_sha(self, sha):
        """Lookup a Git sha in the database.

        :param sha: Git object sha
        :return: (type, type_data) with type_data:
            commit: revid, tree sha
            blob: fileid, revid
            tree: fileid, revid
        """
        if len(sha) == 40:
            sha = hex_to_sha(sha)
        value = self.db[b'git\x00' + sha]
        for data in value.splitlines():
            data = data.split(b'\x00')
            type_name = data[0].decode('ascii')
            if type_name == 'commit':
                if len(data) == 3:
                    yield (type_name, (data[1], data[2], {}))
                else:
                    yield (type_name, (data[1], data[2], {'testament3-sha1': data[3]}))
            elif type_name in ('tree', 'blob'):
                yield (type_name, tuple(data[1:]))
            else:
                raise AssertionError('unknown type %r' % type_name)

    def missing_revisions(self, revids):
        ret = set()
        for revid in revids:
            if self.db.get(b'commit\x00' + revid) is None:
                ret.add(revid)
        return ret

    def _keys(self):
        return self.db.keys()

    def revids(self):
        """List the revision ids known."""
        for key in self._keys():
            if key.startswith(b'commit\x00'):
                yield key[7:]

    def sha1s(self):
        """List the SHA1s."""
        for key in self._keys():
            if key.startswith(b'git\x00'):
                yield sha_to_hex(key[4:])