import sys
from rdkit.Dbase import DbInfo
class DbResultSet(DbResultBase):
    """ Only supports forward iteration """

    def __init__(self, *args, **kwargs):
        DbResultBase.__init__(self, *args, **kwargs)
        self.seen = []
        self._stopped = 0

    def Reset(self):
        self._stopped = 0
        DbResultBase.Reset(self)

    def next(self):
        if self._stopped:
            raise StopIteration
        r = None
        while r is None:
            r = self.cursor.fetchone()
            if not r:
                self._stopped = 1
                raise StopIteration
            if self.transform is not None:
                r = self.transform(r)
            if self.removeDups >= 0:
                v = r[self.removeDups]
                if v in self.seen:
                    r = None
                else:
                    self.seen.append(v)
        return r
    __next__ = next