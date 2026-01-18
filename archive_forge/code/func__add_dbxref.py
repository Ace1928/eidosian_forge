from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _add_dbxref(self, dbname, accession, version):
    """Insert a dbxref and return its id (PRIVATE)."""
    self.adaptor.execute('INSERT INTO dbxref(dbname, accession, version) VALUES (%s, %s, %s)', (dbname, accession, version))
    return self.adaptor.last_id('dbxref')