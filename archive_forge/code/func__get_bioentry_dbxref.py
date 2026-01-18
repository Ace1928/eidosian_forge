from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_bioentry_dbxref(self, bioentry_id, dbxref_id, rank):
    """Get pre-existing db-xref, or create and return it (PRIVATE).

        Check for a pre-existing bioentry_dbxref entry with the passed
        seqfeature_id and dbxref_id.  If one does not exist, insert new
        data
        """
    sql = 'SELECT bioentry_id, dbxref_id FROM bioentry_dbxref WHERE bioentry_id = %s AND dbxref_id = %s'
    result = self.adaptor.execute_and_fetch_col0(sql, (bioentry_id, dbxref_id))
    if result:
        return result
    return self._add_bioentry_dbxref(bioentry_id, dbxref_id, rank)