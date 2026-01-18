import codecs
from gitdb.db.base import (
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