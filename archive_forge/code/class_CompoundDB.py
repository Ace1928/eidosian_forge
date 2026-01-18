from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
class CompoundDB(ObjectDBR, LazyMixin, CachingDB):
    """A database which delegates calls to sub-databases.

    Databases are stored in the lazy-loaded _dbs attribute.
    Define _set_cache_ to update it with your databases"""

    def _set_cache_(self, attr):
        if attr == '_dbs':
            self._dbs = list()
        elif attr == '_db_cache':
            self._db_cache = dict()
        else:
            super()._set_cache_(attr)

    def _db_query(self, sha):
        """:return: database containing the given 20 byte sha
        :raise BadObject:"""
        try:
            return self._db_cache[sha]
        except KeyError:
            pass
        for db in self._dbs:
            if db.has_object(sha):
                self._db_cache[sha] = db
                return db
        raise BadObject(sha)

    def has_object(self, sha):
        try:
            self._db_query(sha)
            return True
        except BadObject:
            return False

    def info(self, sha):
        return self._db_query(sha).info(sha)

    def stream(self, sha):
        return self._db_query(sha).stream(sha)

    def size(self):
        """:return: total size of all contained databases"""
        return reduce(lambda x, y: x + y, (db.size() for db in self._dbs), 0)

    def sha_iter(self):
        return chain(*(db.sha_iter() for db in self._dbs))

    def databases(self):
        """:return: tuple of database instances we use for lookups"""
        return tuple(self._dbs)

    def update_cache(self, force=False):
        self._db_cache.clear()
        stat = False
        for db in self._dbs:
            if isinstance(db, CachingDB):
                stat |= db.update_cache(force)
        return stat

    def partial_to_complete_sha_hex(self, partial_hexsha):
        """
        :return: 20 byte binary sha1 from the given less-than-40 byte hexsha (bytes or str)
        :param partial_hexsha: hexsha with less than 40 byte
        :raise AmbiguousObjectName: """
        databases = list()
        _databases_recursive(self, databases)
        partial_hexsha = force_text(partial_hexsha)
        len_partial_hexsha = len(partial_hexsha)
        if len_partial_hexsha % 2 != 0:
            partial_binsha = hex_to_bin(partial_hexsha + '0')
        else:
            partial_binsha = hex_to_bin(partial_hexsha)
        candidate = None
        for db in databases:
            full_bin_sha = None
            try:
                if hasattr(db, 'partial_to_complete_sha_hex'):
                    full_bin_sha = db.partial_to_complete_sha_hex(partial_hexsha)
                else:
                    full_bin_sha = db.partial_to_complete_sha(partial_binsha, len_partial_hexsha)
            except BadObject:
                continue
            if full_bin_sha:
                if candidate and candidate != full_bin_sha:
                    raise AmbiguousObjectName(partial_hexsha)
                candidate = full_bin_sha
        if not candidate:
            raise BadObject(partial_binsha)
        return candidate