from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_dbxref_id(self, db, accession):
    """Get DB cross-reference for accession (PRIVATE).

        Arguments:
         - db - String, the name of the external database containing
           the accession number
         - accession - String, the accession of the dbxref data

        Finds and returns the dbxref_id for the passed data.  The method
        attempts to find an existing record first, and inserts the data
        if there is no record.
        """
    sql = 'SELECT dbxref_id FROM dbxref WHERE dbname = %s AND accession = %s'
    dbxref_id = self.adaptor.execute_and_fetch_col0(sql, (db, accession))
    if dbxref_id:
        return dbxref_id[0]
    return self._add_dbxref(db, accession, 0)