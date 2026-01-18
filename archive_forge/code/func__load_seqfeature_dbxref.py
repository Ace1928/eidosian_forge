from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_seqfeature_dbxref(self, dbxrefs, seqfeature_id):
    """Add SeqFeature's DB cross-references to the database (PRIVATE).

        Arguments:
         - dbxrefs - List, dbxref data from the source file in the
           format <database>:<accession>
         - seqfeature_id - Int, the identifier for the seqfeature in the
           seqfeature table

        Insert dbxref qualifier data for a seqfeature into the
        seqfeature_dbxref and, if required, dbxref tables.
        The dbxref_id qualifier/value sets go into the dbxref table
        as dbname, accession, version tuples, with dbxref.dbxref_id
        being automatically assigned, and into the seqfeature_dbxref
        table as seqfeature_id, dbxref_id, and rank tuples.
        """
    for rank, value in enumerate(dbxrefs):
        try:
            dbxref_data = value.replace(' ', '').replace('\n', '').split(':')
            db = dbxref_data[0]
            accessions = dbxref_data[1:]
        except Exception:
            raise ValueError(f"Parsing of db_xref failed: '{value}'") from None
        for accession in accessions:
            dbxref_id = self._get_dbxref_id(db, accession)
            self._get_seqfeature_dbxref(seqfeature_id, dbxref_id, rank + 1)