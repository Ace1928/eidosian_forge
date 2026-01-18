from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _add_seqfeature_dbxref(self, seqfeature_id, dbxref_id, rank):
    """Add DB cross-reference (PRIVATE).

        Insert a seqfeature_dbxref row and return the seqfeature_id and
        dbxref_id
        """
    sql = 'INSERT INTO seqfeature_dbxref (seqfeature_id, dbxref_id, "rank") VALUES(%s, %s, %s)'
    self.adaptor.execute(sql, (seqfeature_id, dbxref_id, rank))
    return (seqfeature_id, dbxref_id)