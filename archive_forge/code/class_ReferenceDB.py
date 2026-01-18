import codecs
from gitdb.db.base import (
class ReferenceDB(CompoundDB):
    """A database consisting of database referred to in a file"""
    ObjectDBCls = None

    def __init__(self, ref_file):
        super().__init__()
        self._ref_file = ref_file

    def _set_cache_(self, attr):
        if attr == '_dbs':
            self._dbs = list()
            self._update_dbs_from_ref_file()
        else:
            super()._set_cache_(attr)

    def _update_dbs_from_ref_file(self):
        dbcls = self.ObjectDBCls
        if dbcls is None:
            from gitdb.db.git import GitDB
            dbcls = GitDB
        ref_paths = list()
        try:
            with codecs.open(self._ref_file, 'r', encoding='utf-8') as f:
                ref_paths = [l.strip() for l in f]
        except OSError:
            pass
        ref_paths_set = set(ref_paths)
        cur_ref_paths_set = {db.root_path() for db in self._dbs}
        for path in cur_ref_paths_set - ref_paths_set:
            for i, db in enumerate(self._dbs[:]):
                if db.root_path() == path:
                    del self._dbs[i]
                    continue
        added_paths = sorted(ref_paths_set - cur_ref_paths_set, key=lambda p: ref_paths.index(p))
        for path in added_paths:
            try:
                db = dbcls(path)
                if isinstance(db, CompoundDB):
                    db.databases()
                self._dbs.append(db)
            except Exception:
                pass

    def update_cache(self, force=False):
        self._update_dbs_from_ref_file()
        return super().update_cache(force)