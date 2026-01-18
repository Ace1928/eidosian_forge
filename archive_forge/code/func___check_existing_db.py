import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
def __check_existing_db(self):
    """Perform basic sanity checks upon loading an existing index (PRIVATE)."""
    try:
        idx_version = int(self._con.execute("SELECT value FROM meta_data WHERE key = 'version'").fetchone()[0])
        if idx_version != MAFINDEX_VERSION:
            msg = '\n'.join(['Index version (%s) incompatible with this version of MafIndex' % idx_version, 'You might erase the existing index %s for it to be rebuilt.' % self._index_filename])
            raise ValueError(msg)
        filename = self._con.execute("SELECT value FROM meta_data WHERE key = 'filename'").fetchone()[0]
        if os.path.isabs(filename):
            tmp_mafpath = filename
        else:
            tmp_mafpath = os.path.join(self._relative_path, filename.replace('/', os.path.sep))
        if tmp_mafpath != os.path.abspath(self._maf_file):
            raise ValueError(f'Index uses a different file ({filename} != {self._maf_file})')
        db_target = self._con.execute("SELECT value FROM meta_data WHERE key = 'target_seqname'").fetchone()[0]
        if db_target != self._target_seqname:
            raise ValueError('Provided database indexed for %s, expected %s' % (db_target, self._target_seqname))
        record_count = int(self._con.execute("SELECT value FROM meta_data WHERE key = 'record_count'").fetchone()[0])
        if record_count == -1:
            raise ValueError('Unfinished/partial database provided')
        records_found = int(self._con.execute('SELECT COUNT(*) FROM offset_data').fetchone()[0])
        if records_found != record_count:
            raise ValueError('Expected %s records, found %s.  Corrupt index?' % (record_count, records_found))
        return records_found
    except (dbapi2.OperationalError, dbapi2.DatabaseError) as err:
        raise ValueError(f'Problem with SQLite database: {err}') from None