import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
def __make_new_index(self):
    """Read MAF file and generate SQLite index (PRIVATE)."""
    self._con.execute('CREATE TABLE meta_data (key TEXT, value TEXT);')
    self._con.execute('INSERT INTO meta_data (key, value) VALUES (?, ?);', ('version', MAFINDEX_VERSION))
    self._con.execute("INSERT INTO meta_data (key, value) VALUES ('record_count', -1);")
    self._con.execute('INSERT INTO meta_data (key, value) VALUES (?, ?);', ('target_seqname', self._target_seqname))
    if not os.path.isabs(self._maf_file) and (not os.path.isabs(self._index_filename)):
        mafpath = os.path.relpath(self._maf_file, self._relative_path).replace(os.path.sep, '/')
    elif (os.path.dirname(os.path.abspath(self._maf_file)) + os.path.sep).startswith(self._relative_path + os.path.sep):
        mafpath = os.path.relpath(self._maf_file, self._relative_path).replace(os.path.sep, '/')
    else:
        mafpath = os.path.abspath(self._maf_file)
    self._con.execute('INSERT INTO meta_data (key, value) VALUES (?, ?);', ('filename', mafpath))
    self._con.execute('CREATE TABLE offset_data (bin INTEGER, start INTEGER, end INTEGER, offset INTEGER);')
    insert_count = 0
    mafindex_func = self.__maf_indexer()
    while True:
        batch = list(islice(mafindex_func, 100))
        if not batch:
            break
        self._con.executemany('INSERT INTO offset_data (bin, start, end, offset) VALUES (?,?,?,?);', batch)
        self._con.commit()
        insert_count += len(batch)
    self._con.execute('CREATE INDEX IF NOT EXISTS bin_index ON offset_data(bin);')
    self._con.execute('CREATE INDEX IF NOT EXISTS start_index ON offset_data(start);')
    self._con.execute('CREATE INDEX IF NOT EXISTS end_index ON offset_data(end);')
    self._con.execute(f"UPDATE meta_data SET value = '{insert_count}' WHERE key = 'record_count'")
    self._con.commit()
    return insert_count