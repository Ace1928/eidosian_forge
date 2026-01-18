from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _add_bioentry_dbxref(self, bioentry_id, dbxref_id, rank):
    """Insert a bioentry_dbxref row (PRIVATE).

        Returns the seqfeature_id and dbxref_id (PRIVATE).
        """
    sql = 'INSERT INTO bioentry_dbxref (bioentry_id,dbxref_id,"rank") VALUES (%s, %s, %s)'
    self.adaptor.execute(sql, (bioentry_id, dbxref_id, rank))
    return (bioentry_id, dbxref_id)