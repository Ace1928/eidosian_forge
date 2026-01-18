import os.path
import pickle
import sys
from rdkit import RDConfig
from rdkit.VLib.Supply import SupplyNode
class _dataSeq(_lazyDataSeq):

    def __init__(self, cursor, cmd, pickleCol=1, depickle=1):
        self.cursor = cursor
        self.cmd = cmd
        self.res = None
        self.rowCount = -1
        self.idx = 0
        self._pickleCol = pickleCol
        self._depickle = depickle

    def __iter__(self):
        self.cursor.execute(self.cmd)
        self._first = self.cursor.fetchone()
        self._validate()
        self.res = self.cursor.conn.conn.query('fetch all from "%s"' % self.cursor.name)
        self.rowCount = self.res.ntuples + 1
        self.idx = 0
        if self.res.nfields < 2:
            raise ValueError('bad query result' % str(res))
        return self

    def next(self):
        if self.idx >= self.rowCount:
            raise StopIteration
        fp = self[self.idx]
        self.idx += 1
        return fp

    def __len__(self):
        return self.rowCount

    def __getitem__(self, idx):
        if self.res is None:
            self.cursor.execute(self.cmd)
            self._first = self.cursor.fetchone()
            self._validate()
            self.res = self.cursor.conn.conn.query('fetch all from "%s"' % self.cursor.name)
            self.rowCount = self.res.ntuples + 1
            self.idx = 0
            if self.res.nfields < 2:
                raise ValueError('bad query result' % str(res))
        if idx < 0:
            idx = self.rowCount + idx
        if idx < 0 or (idx >= 0 and idx >= self.rowCount):
            raise IndexError
        if idx == 0:
            val = str(self._first[self._pickleCol])
            t = list(self._first)
        else:
            val = self.res.getvalue(self.idx - 1, self._pickleCol)
            t = [self.res.getvalue(self.idx - 1, x) for x in range(self.res.nfields)]
        if self._depickle:
            try:
                fp = pickle.loads(val)
            except Exception:
                import logging
                del t[self._pickleCol]
                logging.exception('Depickling failure in row: %s' % str(t))
                raise
            del t[self._pickleCol]
            fp._fieldsFromDb = t
        else:
            fp = t
        return fp