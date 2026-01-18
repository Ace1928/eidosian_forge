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
class SqliteGitShaMap(GitShaMap):
    """Bazaar GIT Sha map that uses a sqlite database for storage."""

    def __init__(self, path=None):
        import sqlite3
        self.path = path
        if path is None:
            self.db = sqlite3.connect(':memory:')
        else:
            if path not in mapdbs():
                mapdbs()[path] = sqlite3.connect(path)
            self.db = mapdbs()[path]
        self.db.text_factory = str
        self.db.executescript('\n        create table if not exists commits(\n            sha1 text not null check(length(sha1) == 40),\n            revid text not null,\n            tree_sha text not null check(length(tree_sha) == 40)\n        );\n        create index if not exists commit_sha1 on commits(sha1);\n        create unique index if not exists commit_revid on commits(revid);\n        create table if not exists blobs(\n            sha1 text not null check(length(sha1) == 40),\n            fileid text not null,\n            revid text not null\n        );\n        create index if not exists blobs_sha1 on blobs(sha1);\n        create unique index if not exists blobs_fileid_revid on blobs(\n            fileid, revid);\n        create table if not exists trees(\n            sha1 text unique not null check(length(sha1) == 40),\n            fileid text not null,\n            revid text not null\n        );\n        create unique index if not exists trees_sha1 on trees(sha1);\n        create unique index if not exists trees_fileid_revid on trees(\n            fileid, revid);\n')
        try:
            self.db.executescript('ALTER TABLE commits ADD testament3_sha1 TEXT;')
        except sqlite3.OperationalError:
            pass

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.path)

    def lookup_commit(self, revid):
        cursor = self.db.execute('select sha1 from commits where revid = ?', (revid,))
        row = cursor.fetchone()
        if row is not None:
            return row[0]
        raise KeyError

    def commit_write_group(self):
        self.db.commit()

    def lookup_blob_id(self, fileid, revision):
        row = self.db.execute('select sha1 from blobs where fileid = ? and revid = ?', (fileid, revision)).fetchone()
        if row is not None:
            return row[0]
        raise KeyError(fileid)

    def lookup_tree_id(self, fileid, revision):
        row = self.db.execute('select sha1 from trees where fileid = ? and revid = ?', (fileid, revision)).fetchone()
        if row is not None:
            return row[0]
        raise KeyError(fileid)

    def lookup_git_sha(self, sha):
        """Lookup a Git sha in the database.

        :param sha: Git object sha
        :return: (type, type_data) with type_data:
            commit: revid, tree sha, verifiers
            tree: fileid, revid
            blob: fileid, revid
        """
        found = False
        cursor = self.db.execute('select revid, tree_sha, testament3_sha1 from commits where sha1 = ?', (sha,))
        for row in cursor.fetchall():
            found = True
            if row[2] is not None:
                verifiers = {'testament3-sha1': row[2]}
            else:
                verifiers = {}
            yield ('commit', (row[0], row[1], verifiers))
        cursor = self.db.execute('select fileid, revid from blobs where sha1 = ?', (sha,))
        for row in cursor.fetchall():
            found = True
            yield ('blob', row)
        cursor = self.db.execute('select fileid, revid from trees where sha1 = ?', (sha,))
        for row in cursor.fetchall():
            found = True
            yield ('tree', row)
        if not found:
            raise KeyError(sha)

    def revids(self):
        """List the revision ids known."""
        return (row for row, in self.db.execute('select revid from commits'))

    def sha1s(self):
        """List the SHA1s."""
        for table in ('blobs', 'commits', 'trees'):
            for sha, in self.db.execute('select sha1 from %s' % table):
                yield sha.encode('ascii')